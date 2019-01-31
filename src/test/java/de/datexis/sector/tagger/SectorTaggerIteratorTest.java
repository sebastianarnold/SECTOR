package de.datexis.sector.tagger;

import de.datexis.sector.tagger.SectorTaggerIterator;
import de.datexis.sector.tagger.SectorTagger;
import de.datexis.sector.tagger.DocumentSentenceIterator;
import com.google.common.collect.Lists;

import de.datexis.common.Resource;
import de.datexis.encoder.impl.BagOfWordsEncoder;
import de.datexis.encoder.impl.DummyEncoder;
import de.datexis.encoder.impl.StructureEncoder;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.sector.encoder.HeadingEncoder;
import de.datexis.sector.encoder.HeadingTag;
import de.datexis.sector.reader.WikiSectionReader;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;
import static org.hamcrest.Matchers.*;
import static org.mockito.Mockito.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.Collection;
import java.util.List;


public class SectorTaggerIteratorTest {

  private SectorTagger sectorTagger;
  private List<Document> documents;
  private Dataset train;

  @Before
  public void setup() throws IOException {
    Resource testDataPath = Resource.fromJAR("testdata").resolve("en_disease_dementia.json");
    train = WikiSectionReader.readDatasetFromJSON(testDataPath);
    documents = Lists.newArrayList(train.getDocuments());

    sectorTagger = new SectorTagger();
    BagOfWordsEncoder bagEncoder = new BagOfWordsEncoder();
    bagEncoder.trainModel(documents);

    HeadingEncoder headingEncoder = new HeadingEncoder();
    headingEncoder.trainModel(documents);
    HeadingTag.Factory headingTags = new HeadingTag.Factory(headingEncoder);
    for(Document doc : train.getDocuments()) headingTags.attachFromSectionAnnotations(doc, Annotation.Source.GOLD);

    StructureEncoder flagEncoder = new StructureEncoder();
    flagEncoder.trainModel(documents);
    sectorTagger.setInputEncoders(bagEncoder, new DummyEncoder(), flagEncoder);
    sectorTagger.setTargetEncoder(headingEncoder);
  }

  @Test
  public void bagOfWordsEncodingShouldBeEqualToOldImplementation() {
    MultiDataSet expected = generateExpectedEncoding(documents, sectorTagger,
                                                     documents.get(0).countSentences());

    DocumentSentenceIterator.DocumentBatch documentBatch = setUpDocumentBatchMock();
    MultiDataSet actual = generateActualEncoding(documentBatch);
    
    INDArray actualBagOfWords = actual.getFeatures(0);
    INDArray expectedBagOfWords = expected.getFeatures(0);

    assertThat(actualBagOfWords, is(equalTo(expectedBagOfWords)));
  }  
  
  @Test
  public void embeddingEncodingShouldBeEqualToOldImplementation() {
    MultiDataSet expected = generateExpectedEncoding(documents, sectorTagger,
                                                     documents.get(0).countSentences());

    DocumentSentenceIterator.DocumentBatch documentBatch = setUpDocumentBatchMock();
    MultiDataSet actual = generateActualEncoding(documentBatch);
    
    INDArray actualEmbedding = actual.getFeatures(1);
    INDArray expectedEmbedding = expected.getFeatures(1);

    assertThat(actualEmbedding, is(equalTo(expectedEmbedding)));
  }  
  
  @Test
  public void flagEncodingShouldBeEqualToOldImplementation() {
    MultiDataSet expected = generateExpectedEncoding(documents, sectorTagger,
                                                     documents.get(0).countSentences());

    DocumentSentenceIterator.DocumentBatch documentBatch = setUpDocumentBatchMock();
    MultiDataSet actual = generateActualEncoding(documentBatch);
    
    INDArray actualFlagEncoding = actual.getFeatures(2);
    INDArray expectedFlagEncoding = expected.getFeatures(2);
    
    assertThat(actualFlagEncoding, is(equalTo(expectedFlagEncoding)));
  }

  private MultiDataSet generateActualEncoding(DocumentSentenceIterator.DocumentBatch documentBatch) {
    SectorTaggerIterator sectorTaggerIterator = new SectorTaggerIterator(DocumentSentenceIterator.Stage.TRAIN, train, sectorTagger, 1, false, false);
    return sectorTaggerIterator.generateDataSet(documentBatch);
  }
 

  private DocumentSentenceIterator.DocumentBatch setUpDocumentBatchMock() {
    DocumentSentenceIterator.DocumentBatch documentBatch = mock(DocumentSentenceIterator.DocumentBatch.class);
    documentBatch.size = documents.size();
    documentBatch.maxDocLength = documents.get(0).countSentences();
    documentBatch.docs = documents;
    org.nd4j.linalg.dataset.MultiDataSet multiDataSet = new org.nd4j.linalg.dataset.MultiDataSet();
    documentBatch.dataset = multiDataSet;
    return documentBatch;
  }

  private MultiDataSet generateExpectedEncoding(List<Document> batch, SectorTagger tagger, int maxDocLength) {
    // inputs
    INDArray bag = Nd4j.zeros(batch.size(), tagger.bagEncoder.getEmbeddingVectorSize(), maxDocLength);
    INDArray emb = Nd4j.zeros(batch.size(), tagger.embEncoder.getEmbeddingVectorSize(), maxDocLength);
    INDArray flag = Nd4j.zeros(batch.size(), tagger.flagEncoder.getEmbeddingVectorSize(), maxDocLength);
    INDArray inputMask = Nd4j.zeros(batch.size(), maxDocLength);
    // targets
    INDArray targets = Nd4j.zeros(batch.size(), tagger.targetEncoder.getEmbeddingVectorSize(), maxDocLength);
    INDArray labelMask = Nd4j.zeros(batch.size(), maxDocLength);
    // empty data set
    MultiDataSet result = new org.nd4j.linalg.dataset.MultiDataSet(
      new INDArray[] {bag, emb, flag},
      new INDArray[] {targets},
      new INDArray[] {inputMask, inputMask, inputMask},
      new INDArray[] {labelMask}
    );

    // attach all encodings to the document on Sentence level
    tagger.bagEncoder.encodeEach((Collection<Document>) batch, Sentence.class);
    tagger.embEncoder.encodeEach((Collection<Document>) batch, Sentence.class);
    tagger.flagEncoder.encodeEach((Collection<Document>) batch, Sentence.class);


    Document example;
    for(int batchNum = 0; batchNum < batch.size(); batchNum++) {

      example = batch.get(batchNum);
      int t = 0;

      // generate input matrix by going through each Sentence in the Document
      for(Sentence s : example.getSentences()) {

        // set masks
        inputMask.put(batchNum, t, 1); // mark this sentence as used
        labelMask.put(batchNum, t, 1); // mark this sentence as used

        // set inputs ==========================================================
        bag.getRow(batchNum).getColumn(t).assign(s.getVector(tagger.bagEncoder.getClass()));
        emb.getRow(batchNum).getColumn(t).assign(s.getVector(tagger.embEncoder.getClass()));
        flag.getRow(batchNum).getColumn(t).assign(s.getVector(tagger.flagEncoder.getClass()));

        // remove attached encodings
        s.clearVectors(tagger.bagEncoder.getClass());
        s.clearVectors(tagger.embEncoder.getClass());
        s.clearVectors(tagger.flagEncoder.getClass());

        t++;
      }
    }

    Nd4j.getExecutioner().commit();
    return result;
  }


}