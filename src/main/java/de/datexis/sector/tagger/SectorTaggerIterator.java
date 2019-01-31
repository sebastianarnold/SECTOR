package de.datexis.sector.tagger;

import com.google.common.collect.Lists;
import de.datexis.encoder.Encoder;
import de.datexis.encoder.EncoderSet;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.model.Token;
import de.datexis.sector.encoder.ClassEncoder;
import de.datexis.sector.encoder.ClassTag;
import de.datexis.sector.encoder.HeadingEncoder;
import de.datexis.sector.encoder.HeadingTag;

import de.datexis.sector.model.SectionAnnotation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

import org.nd4j.linalg.indexing.INDArrayIndex;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**
 * Iterates through a Dataset with Document-Level Batches of Sentences
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class SectorTaggerIterator extends DocumentSentenceIterator {

  protected EncoderSet inputEncoders, targetEncoders;
  protected SectorTagger tagger;
  protected boolean requireSubsampling;
  
  public SectorTaggerIterator(Stage stage, Dataset dataset, SectorTagger tagger, int batchSize, boolean randomize, boolean useMultiClassLabels) {
    this(stage, dataset.getDocuments(), tagger, batchSize, randomize, useMultiClassLabels);
  }
  
  public SectorTaggerIterator(Stage stage, Collection<Document> docs, SectorTagger tagger, int batchSize, boolean randomize, boolean requireSubsampling) {
    this(stage, docs, tagger, -1, batchSize, randomize, requireSubsampling);
  }
    
  public SectorTaggerIterator(Stage stage, Collection<Document> docs, SectorTagger tagger, int numExamples, int batchSize, boolean randomize, boolean requireSubsampling) {
    this(stage, docs, tagger, numExamples, -1, batchSize, randomize, requireSubsampling);
  }
  
  public SectorTaggerIterator(Stage stage, Collection<Document> docs, SectorTagger tagger, int numExamples, int maxTimeSeriesLength, int batchSize, boolean randomize, boolean requireSubsampling) {
    super(stage, docs, numExamples, maxTimeSeriesLength, batchSize, randomize);
    log = LoggerFactory.getLogger(SectorTaggerIterator.class);
    this.tagger = tagger;
    this.inputEncoders = new EncoderSet(tagger.bagEncoder, tagger.embEncoder, tagger.flagEncoder);
    this.targetEncoders = new EncoderSet(tagger.targetEncoder);
    this.requireSubsampling = requireSubsampling;
    reset();
  }
  
  @Override
  public boolean asyncSupported() {
    return true;
  }
  
  @Override
  public MultiDataSet generateDataSet(DocumentBatch batch) {

    // input encodings
    INDArray inputMask = createMask(batch.docs, batch.maxDocLength, Sentence.class);
    //INDArray labelMask = createMask(batch.docs, batch.maxDocLength, Sentence.class); // same as input mask
    // return all encodings on Sentence level
    INDArray bag = tagger.bagEncoder.encodeMatrix(batch.docs, batch.maxDocLength, Sentence.class); 
    INDArray emb = tagger.embEncoder.encodeMatrix(batch.docs, batch.maxDocLength, Sentence.class);
    INDArray flag = tagger.flagEncoder.encodeMatrix(batch.docs, batch.maxDocLength, Sentence.class);

    // target encodings
    INDArray targets;
    if(stage.equals(Stage.TRAIN) || stage.equals(Stage.TEST)) targets = encodeTarget(batch.docs, batch.maxDocLength, Sentence.class);
    else targets = Nd4j.zeros(batch.size, tagger.targetEncoder.getEmbeddingVectorSize(), batch.maxDocLength);
    
    return new org.nd4j.linalg.dataset.MultiDataSet(
      new INDArray[]{bag, emb, flag},
      new INDArray[]{targets, targets},
      new INDArray[]{inputMask, inputMask, inputMask},
      new INDArray[]{inputMask, inputMask}
    );
  }
  
  public INDArray createMask(List<Document> input, int maxTimeSteps, Class<? extends Span> timeStepClass) {

    INDArray mask = Nd4j.zeros(input.size(), maxTimeSteps, 'f');
    Document example;

    for(int batchIndex = 0; batchIndex < input.size(); batchIndex++) {
      example = input.get(batchIndex);

      int spanCount = 0;
      if(timeStepClass == Token.class) spanCount = example.countTokens();
      else if(timeStepClass == Sentence.class) spanCount = example.countSentences();

      for(int t = 0; t < spanCount && t < maxTimeSteps; t++) {
        mask.putScalar(new int[] {batchIndex, t}, 1);
      }
      
    }
    return mask;
  }
  
  public INDArray encodeTarget(List<Document> input, int maxTimeSteps, Class<? extends Span> timeStepClass) {
    
    INDArray encoding = Nd4j.zeros(input.size(), tagger.targetEncoder.getEmbeddingVectorSize(), maxTimeSteps);
    Document example;

    for(int batchIndex = 0; batchIndex < input.size(); batchIndex++) {
      example = input.get(batchIndex);

      List<? extends Span> spansToEncode = Collections.EMPTY_LIST;
      if(timeStepClass == Token.class) spansToEncode = Lists.newArrayList(example.getTokens());
      else if(timeStepClass == Sentence.class) spansToEncode = Lists.newArrayList(example.getSentences());

      List<SectionAnnotation> anns = example
        .streamAnnotations(Annotation.Source.GOLD, SectionAnnotation.class)
        .sorted()
        .collect(Collectors.toList());

      Iterator<SectionAnnotation> it = anns.iterator();
      if(!it.hasNext()) return encoding; // no annotations
      SectionAnnotation ann = it.next();
      INDArray vec = encodeTag(tagger.targetEncoder, ann);

      for(int t = 0; t < spansToEncode.size() && t < maxTimeSteps; t++) {
        Span s = spansToEncode.get(t);
        if(s.getBegin() >= ann.getEnd() && it.hasNext()) {
          // encode the next section
          ann = it.next();
          vec = encodeTag(tagger.targetEncoder, ann);
        }
        encoding.getRow(batchIndex).getColumn(t).assign(vec.dup()); // this one is faster
      }
      
    }
    return encoding;
  }

  protected INDArray encodeTag(Encoder enc, SectionAnnotation ann) {
    if(enc instanceof HeadingEncoder) {
      if(requireSubsampling) return ((HeadingEncoder) enc).encodeSubsampled(ann.getSectionHeading());
      else return ((HeadingEncoder) enc).encode(ann.getSectionHeading());
    } else if(enc instanceof ClassEncoder) {
      return ((ClassEncoder) enc).encode(ann.getSectionLabel());
    } else return Nd4j.create(1);
  }
  
}
