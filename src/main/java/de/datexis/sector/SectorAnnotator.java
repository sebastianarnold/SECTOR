package de.datexis.sector;

import de.datexis.sector.model.SectionAnnotation;
import de.datexis.annotator.Annotator;
import de.datexis.annotator.AnnotatorComponent;
import de.datexis.common.WordHelpers;
import de.datexis.encoder.Encoder;
import de.datexis.encoder.LookupCacheEncoder;
import de.datexis.sector.encoder.ClassEncoder;
import de.datexis.model.Annotation;
import de.datexis.model.Annotation.Source;
import de.datexis.tagger.Tagger;
import de.datexis.model.Document;
import de.datexis.model.Dataset;
import de.datexis.model.Sentence;
import de.datexis.sector.encoder.ClassTag;
import de.datexis.sector.encoder.HeadingEncoder;
import de.datexis.sector.encoder.HeadingTag;
import de.datexis.sector.eval.SectorEvaluation;
import de.datexis.sector.tagger.DocumentSentenceIterator;
import de.datexis.sector.tagger.ScoreImprovementMinEpochsTerminationCondition;
import de.datexis.sector.tagger.SectorEncoder;
import de.datexis.sector.tagger.SectorTagger;
import java.util.Collection;
import java.util.Map;
import java.util.stream.Collectors;
import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.dimensionalityreduction.PCA;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.memory.abstracts.DummyWorkspace;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An Annotator that detects sections in a Document and assigns labels. Implementation of:
 * Sebastian Arnold, Rudolf Schneider, Philippe Cudré-Mauroux, Felix A. Gers and Alexander Löser:
 * "SECTOR: A Neural Model for Coherent Topic Segmentation and Classification."
 * Transactions of the Association for Computational Linguistics (2019).
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class SectorAnnotator extends Annotator {

  protected final static Logger log = LoggerFactory.getLogger(SectorAnnotator.class);
  
  public static enum SegmentationMethod {
    NONE, // don't segment, only tag sentences
    GOLD, // use provided gold standard segmentation (perfect case)
    NL, // segment at every newline (will produce too many segments)
    MAX, // segment if top-2 labels change
    EMD, // segmentation based on edge detection on embedding deviation
    BEMD, // segmentation based on edge detection on bidirectional embedding deviation (FW/BW)
    BEMD_FIXED, // use BEMD with provided ground truth number of sections
  };

  /** used for JSON deserialization */
  public SectorAnnotator() {
  }
  
  public SectorAnnotator(Tagger root) {
    super(root);
  }
  
  protected SectorAnnotator(AnnotatorComponent comp) {
    super(comp);
  }
  
  @Override
  public SectorTagger getTagger() {
    return (SectorTagger) super.getTagger();
  }
  
  public LookupCacheEncoder getTargetEncoder() {
    return (LookupCacheEncoder) getTagger().getTargetEncoder();
  }

  /**
   * Annotate given Documents using SECTOR, i.e. attach SectorAnnotator vectors to sentences.
   * This function also attaches SectionAnnotations to each Document by using the BEMD strategy.
   */
  @Override
  public void annotate(Collection<Document> docs) {
    annotate(docs, SegmentationMethod.BEMD);
  }

  /**
   * Annotate given Documents using SECTOR, i.e. attach SectorEncoder vectors to sentences.
   * If a segmentation method is given, also attach SectionAnnotations to each Document.
   */
  public void annotate(Collection<Document> docs, SegmentationMethod segmentation) {
    // use tagger to generate and attach PRED vectors to Sentences
    log.info("Running SECTOR neural net encoding...");
    getTagger().attachVectors(docs, DocumentSentenceIterator.Stage.ENCODE, getTargetEncoder().getClass());
    if(!segmentation.equals(SegmentationMethod.NONE)) segment(docs, segmentation, true);
  }

  /**
   * Attach SectionAnnotations to each Document with a given segmentation strategy.
   * If there are no SectorEncoder vectors attached to sentences yet, please use annotate().
   */
  public void segment(Collection<Document> docs, SegmentationMethod segmentation, boolean mergeSections) {
    // create Annotations and attach vectors
    log.info("Predicting segmentation {}...", segmentation.toString());
    detectSections(docs, segmentation);
    if(mergeSections) {
      // TODO: merge sections
    }
    // attach vectors to annotations
    log.info("Attaching Annotations...");
    for(Document doc : docs) attachVectorsToAnnotations(doc, getTargetEncoder());
    log.info("Segmentation done.");
  }
  
  protected void detectSections(Collection<Document> docs, SegmentationMethod segmentation) {
    WorkspaceMode cMode = getTagger().getNN().getConfiguration().getInferenceWorkspaceMode();
    getTagger().getNN().getConfiguration().setTrainingWorkspaceMode(getTagger().getNN().getConfiguration().getInferenceWorkspaceMode());
    MemoryWorkspace workspace =
            getTagger().getNN().getConfiguration().getTrainingWorkspaceMode() == WorkspaceMode.NONE ? new DummyWorkspace()
                    : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread();

    for(Document doc : docs) {
      try (MemoryWorkspace wsE = workspace.notifyScopeEntered()) {
        switch(segmentation) {
          case GOLD: {
            applySectionsFromGold(doc); 
          } break;
          case MAX: {
            applySectionsFromTargetLabels(doc, getTargetEncoder(), 2); 
          } break;
          case EMD: {
            INDArray mag = detectSectionsFromEmbeddingDeviation(doc);
            applySectionsFromEdges(doc, detectEdges(mag));
          } break;
          case BEMD: {
            INDArray mag = detectSectionsFromBidirectionalEmbeddingDeviation(doc);
            applySectionsFromEdges(doc, detectEdges(mag));
          } break;
          case BEMD_FIXED: {
            INDArray mag = detectSectionsFromBidirectionalEmbeddingDeviation(doc);
            int expectedNumberOfSections = (int) doc.countAnnotations(Source.GOLD);
            applySectionsFromEdges(doc, detectEdges(mag, expectedNumberOfSections));
          } break;
          case NL:
          default: {
            applySectionsFromNewlines(doc);
          }
        }
      }
    }
    
    getTagger().getNN().getConfiguration().setTrainingWorkspaceMode(cMode);
    
  }

  /**
   * Evaluate SECTOR model using a given Dataset. This method will print a result table.
   * @return MAP score for segment-level evaluation.
   */
  public double evaluateModel(Dataset test) {
    return evaluateModel(test, true, true, true);
  }

  /**
   * Evaluate SECTOR model using a given Dataset. This method will print a result table.
   * @param evalSentenceClassification - enable/disable the evaluation of sentence-level classification (P/R scores)
   * @param evalSegmentation - enable/disable the evaluation of text segmentation (Pk/WD scores)
   * @param evalSegmentClassification - enable/disable the evaluation of segment-level classification (P/R scores)
   * @return MAP score for segment-level evaluation.
   */
  public double evaluateModel(Dataset test, boolean evalSentenceClassification, boolean evalSegmentation, boolean evalSegmentClassification)  {
    SectorEvaluation eval;
    if(getTargetEncoder().getClass() == HeadingEncoder.class) {
      HeadingEncoder headings = ((HeadingEncoder)getComponent(HeadingEncoder.ID));
      eval = new SectorEvaluation(test.getName(), Annotation.Source.GOLD, Annotation.Source.PRED, headings);
      // we need tags for sentence-level evaluation
      if(evalSentenceClassification) {
        log.info("Creating tags...");
        removeTags(test.getDocuments(), Annotation.Source.PRED);
        createHeadingTags(test.getDocuments(), Annotation.Source.GOLD, headings);
        createHeadingTags(test.getDocuments(), Annotation.Source.PRED, headings);
      }
    } else if(getTargetEncoder().getClass() == ClassEncoder.class) {
      ClassEncoder classes = ((ClassEncoder)getComponent(ClassEncoder.ID));
      eval = new SectorEvaluation(test.getName(), Annotation.Source.GOLD, Annotation.Source.PRED, classes);
      // we need tags for sentence-level evaluation
      if(evalSentenceClassification) {
        log.info("Creating tags...");
        removeTags(test.getDocuments(), Annotation.Source.PRED);
        createClassTags(test.getDocuments(), Annotation.Source.GOLD, classes);
        createClassTags(test.getDocuments(), Annotation.Source.PRED, classes);
      }
    } else {
      throw new IllegalArgumentException("Target encoder has no evaluation: " + getTargetEncoder().getClass().toString());
    }
    // calculate and print scores
    eval.withSentenceClassEvaluation(evalSentenceClassification)
        .withSegmentationEvaluation(evalSegmentClassification)
        .withSegmentClassEvaluation(evalSegmentation)
        .calculateScores(test);
    getTagger().appendTestLog(eval.printDatasetStats(test));
    getTagger().appendTestLog(eval.printEvaluationStats());
    getTagger().appendTestLog(eval.printSingleClassStats());
    return eval.getScore();
  }

  /**
   * Train a SECTOR model with configured number of epochs.
   */
  public void trainModel(Dataset train) {
    provenance.setDataset(train.getName());
    provenance.setLanguage(train.getLanguage());
    getTagger().trainModel(train);
  }

  /**
   * Train a SECTOR model with given fixed number of epochs.
   */
  public void trainModel(Dataset train, int numEpochs) {
    provenance.setDataset(train.getName());
    provenance.setLanguage(train.getLanguage());
    getTagger().trainModel(train, numEpochs);
  }

  /**
   * Train a SECTOR model with early stopping based on MAP score. The best model will be used after this call.
   * @param train training Dataset with GOLD Annotations
   * @param validation validation Dataset with GOLD Annotations
   * @param minEpochs training will not be stopped before this number of epochs (absolute value)
   * @param minEpochsNoImprovement training will be stopped after this number of epochs without a MAP improvement (relative value)
   * @param maxEpochs training will be stopped after this number of epochs (absolute value)
   */
  public void trainModelEarlyStopping(Dataset train, Dataset validation, int minEpochs, int minEpochsNoImprovement, int maxEpochs) {
    EarlyStoppingConfiguration conf = new EarlyStoppingConfiguration.Builder()
        .evaluateEveryNEpochs(1)
        .epochTerminationConditions(new ScoreImprovementMinEpochsTerminationCondition(minEpochs, minEpochsNoImprovement, maxEpochs))
        .saveLastModel(false)
        .build();
    // train tagger
    EarlyStoppingResult<ComputationGraph> result = getTagger().trainModel(train, validation, conf);
    getTagger().appendTrainLog("Training complete " + result.toString());
  }

  /**
   * Create heading tags that are only required for Sentence-level evaluation
   */
  private void createHeadingTags(Iterable<Document> docs, Annotation.Source source, HeadingEncoder headings) {
    HeadingTag.Factory headingTags = new HeadingTag.Factory(headings);
    for(Document doc : docs) {
      if(!doc.isTagAvaliable(source, HeadingTag.class)) {
        if(source.equals(Annotation.Source.GOLD)) headingTags.attachFromSectionAnnotations(doc, source);
        else if(source.equals(Annotation.Source.PRED)) headingTags.attachFromSentenceVectors(doc, HeadingEncoder.class, source);
      }
    }
  }

  /**
   * Create class tags that are only required for Sentence-level evaluation
   */
  private void createClassTags(Iterable<Document> docs, Annotation.Source source, ClassEncoder classes) {
    ClassTag.Factory classTags = new ClassTag.Factory(classes);
    for(Document doc : docs) {
      if(!doc.isTagAvaliable(source, ClassTag.class)) {
        if(source.equals(Annotation.Source.GOLD)) classTags.attachFromSectionAnnotations(doc, source);
        else if(source.equals(Annotation.Source.PRED)) classTags.attachFromSentenceVectors(doc, ClassEncoder.class, source);
      }
    }
  }

  /**
   * Clear tags that are only required for Sentence-level evaluation
   */
  private static void removeTags(Iterable<Document> docs, Annotation.Source source) {
    for(Document doc : docs) {
      for(Sentence s : doc.getSentences()) {
        s.clearTags(source);
      }
      doc.setTagAvailable(source, HeadingTag.class, false);
      doc.setTagAvailable(source, ClassTag.class, false);
    }
  }
  
  /**
   * Add vectors and class labels for all existing GOLD and PRED annotations.
   */
  protected static void attachVectorsToAnnotations(Document doc, LookupCacheEncoder targetEncoder) {
    // attach GOLD vectors
    for(SectionAnnotation ann : doc.getAnnotations(Annotation.Source.GOLD, SectionAnnotation.class)) {
      if(targetEncoder.getClass() == ClassEncoder.class) {
        INDArray exp = targetEncoder.encode(ann.getSectionLabel());
        ann.putVector(ClassEncoder.class, exp);
      } else if(targetEncoder.getClass() == HeadingEncoder.class) {
        INDArray exp = targetEncoder.encode(ann.getSectionHeading());
        ann.putVector(HeadingEncoder.class, exp);
      }
    }
    // attach PRED vectors and labels from empty Annotations
    for(SectionAnnotation ann : doc.getAnnotations(Annotation.Source.PRED, SectionAnnotation.class)) {
      int count = 0;
      INDArray pred = Nd4j.zeros(targetEncoder.getEmbeddingVectorSize(), 1);
      for(Sentence s : doc.streamSentencesInRange(ann.getBegin(), ann.getEnd(), false).collect(Collectors.toList())) {
        pred.addi(s.getVector(targetEncoder.getClass()));
        count++;
      }
      if(count > 1) pred.divi(count);
      if(targetEncoder.getClass() == ClassEncoder.class) {
        ann.putVector(ClassEncoder.class, pred);
        ann.setSectionLabel(targetEncoder.getNearestNeighbour(pred));
        ann.setConfidence(targetEncoder.getMaxConfidence(pred));
      } else if(targetEncoder.getClass() == HeadingEncoder.class) {
        ann.putVector(HeadingEncoder.class, pred);
        Collection<String> preds = targetEncoder.getNearestNeighbours(pred, 2);
        ann.setSectionHeading(StringUtils.join(preds, "/"));
        ann.setConfidence(targetEncoder.getMaxConfidence(pred));
      }
    }
  }
  
  /**
   * Add PRED SectionAnnotations from provided gold standard segmentation (perfect case)
   */
  private static void applySectionsFromGold(Document doc) {
    SectionAnnotation section = null;
    for(SectionAnnotation ann : doc.getAnnotations(Source.GOLD, SectionAnnotation.class)) {
      section = new SectionAnnotation(Annotation.Source.PRED);
      section.setBegin(ann.getBegin());
      section.setEnd(ann.getEnd());
      doc.addAnnotation(section);
    }
  }
  
  /**
   * Add PRED SectionAnnotation at every newline (will produce too many segments)
   */
  private static void applySectionsFromNewlines(Document doc) {
    SectionAnnotation section = null;
    for(Sentence s : doc.getSentences()) {
      boolean endPar = s.streamTokens().anyMatch(t -> t.getText().equals("*NL*") || t.getText().equals("\n"));
      if(section == null) {
        section = new SectionAnnotation(Annotation.Source.PRED);
        section.setBegin(s.getBegin());
      }
      if(endPar) {
        section.setEnd(s.getEnd());
        doc.addAnnotation(section);
        section = null;
      }
    }
    if(section != null) {
      log.warn("found last sentence without newline");
      section.setEnd(doc.getEnd());
      doc.addAnnotation(section);
      section = null;
    }
  }
  
  /**
   * Add PRED Section Annotation based on sentence-wise output predictions.
   * A new segment will start if top label is not contained in previous top-k labels.
   * @param k - the number of labels to check for change (usually 1-3)
   */
  private static void applySectionsFromTargetLabels(Document doc, LookupCacheEncoder targetEncoder, int k) {
    // start first section
    String lastSection = "";
    INDArray sectionPredictions = Nd4j.create(1, targetEncoder.getEmbeddingVectorSize()).transposei();
    int sectionLength = 0;
    SectionAnnotation section = new SectionAnnotation(Annotation.Source.PRED);
    section.setBegin(doc.getBegin());

    for(Sentence s : doc.getSentences()) {
      INDArray pred = s.getVector(targetEncoder.getClass());
      Collection<String> currentSections = targetEncoder.getNearestNeighbours(pred, k);
      // start new section
      if(!currentSections.contains(lastSection)) {
        if(!lastSection.isEmpty()) doc.addAnnotation(section);
        section = new SectionAnnotation(Annotation.Source.PRED);
        section.setBegin(s.getBegin());
        sectionLength = 0;
        sectionPredictions = Nd4j.create(1, targetEncoder.getEmbeddingVectorSize()).transposei();
      }
      // update current section
      sectionPredictions.addi(pred);
      sectionLength++;
      String currentSection = targetEncoder.getNearestNeighbour(sectionPredictions.div(sectionLength));
      section.setEnd(s.getEnd());
      lastSection = currentSection;
    }

    // add last section
    if(!lastSection.isEmpty()) doc.addAnnotation(section);
  }

  /**
   * Add PRED SectionAnnotations from given edge array.
   */
  private static void applySectionsFromEdges(Document doc, INDArray docEdges) {
    
    // no sentence
    if(doc.countSentences() < 1) {
      log.warn("Empty document");
      return;
    }
    
    // single sentence
    if(docEdges == null || doc.countSentences() < 2) {
      SectionAnnotation section = new SectionAnnotation(Annotation.Source.PRED);
      section.setBegin(doc.getBegin());
      section.setEnd(doc.getEnd());
      doc.addAnnotation(section);
      return;
    }
    
    // start first section
    int sectionLength = 0;
    SectionAnnotation section = new SectionAnnotation(Annotation.Source.PRED);
    section.setBegin(doc.getBegin());

    int t = 0;
    for(Sentence s : doc.getSentences()) {
      // start new section
      if(docEdges.getDouble(t) > 0) {
        if(sectionLength > 0) doc.addAnnotation(section);
        section = new SectionAnnotation(Annotation.Source.PRED);
        section.setBegin(s.getBegin());
        sectionLength = 0;
      }
      // update current section
      sectionLength++;
      section.setEnd(s.getEnd());
      t++;
    }

    // add last section
    if(sectionLength > 0) doc.addAnnotation(section);
    
  }
  
  /**
   * Add PRED SectionAnnotations based on edge detection on embedding deviation.
   */
  private static INDArray detectSectionsFromEmbeddingDeviation(Document doc) {
    
    int PCA_DIMS = 16;
    
    if(doc.countSentences() < 2) return null;
    
    // initialize embedding matrix
    INDArray docEmbs = getEmbeddingMatrix(doc);
    
    INDArray docPCA = pca(docEmbs, PCA_DIMS);
    INDArray docSmooth = gaussianSmooth(docPCA);
    INDArray docMag = deviation(docSmooth);
    
    return docMag;
    
  }
  
  /**
   * Add PRED SectionAnnotations based on edge detection on bidirectional (FW/BW) embedding deviation.
   */
  private static INDArray detectSectionsFromBidirectionalEmbeddingDeviation(Document doc) {
    
    int PCA_DIMS = 16;
    double SMOOTH_FACTOR = 1.5;
    
    if(doc.countSentences() < 1) return null;
    Sentence sent = doc.getSentence(0);
      
    // initialize FW/BW matrices
    long layerSize = sent.getVector("embeddingFW").length();
    INDArray docFW = Nd4j.zeros(doc.countSentences(), layerSize);
    INDArray docBW = Nd4j.zeros(doc.countSentences(), layerSize);
    
    // fill FW/BW matrices
    int t = 0;
    for(Sentence s : doc.getSentences()) {
      docFW.getRow(t).assign(s.getVector("embeddingFW"));
      docBW.getRow(t).assign(s.getVector("embeddingBW"));
      t++;
    }
    
    INDArray docFwPCA = docFW.mmul(PCA.pca_factor(docFW.dup(), PCA_DIMS, false));
    INDArray docBwPCA = docBW.mmul(PCA.pca_factor(docBW.dup(), PCA_DIMS, false));
    // remove first principal components
    INDArray zeros = Nd4j.zeros(docFW.rows(), 1);
    docFwPCA.putColumn(0, zeros);
    docBwPCA.putColumn(0, zeros);
    docFwPCA.putColumn(1, zeros);
    docBwPCA.putColumn(1, zeros);
    INDArray docFwPCAs = gaussianSmooth(docFwPCA, SMOOTH_FACTOR);
    INDArray docBwPCAs = gaussianSmooth(docBwPCA, SMOOTH_FACTOR);
    INDArray docMag = deviation(docFwPCAs, docBwPCAs);
    
    return docMag;
    
  }
  
  /**
   * @return Matrix sentences x layersize that contains target predictions for a Document
   */
  protected static INDArray getLayerMatrix(Document doc, String layerClass) {
    
    Sentence sent = doc.getSentence(0);
    
    // initialize embedding matrix
    long layerSize = sent.getVector(layerClass).length();
    INDArray docWeights = Nd4j.zeros(doc.countSentences(), layerSize);
    
    // fill embedding matrix
    int t = 0;
    for(Sentence s : doc.getSentences()) {
      docWeights.getRow(t++).assign(s.getVector(layerClass));
    }

    return docWeights;
    
  }

  protected static INDArray getLayerMatrix(Document doc, Class layerClass) {
    return getLayerMatrix(doc, layerClass.getCanonicalName());
  }
  
  /**
   * @return Matrix sentences x layersize that contains Sector embeddings for a Document
   */
  protected static INDArray getEmbeddingMatrix(Document doc) {
    return getLayerMatrix(doc, SectorEncoder.class);
  }

  protected static INDArray pca(INDArray m, int dimensions) {
    return m.mmul(PCA.pca_factor(m.dup(), dimensions, true));
  }

  protected static INDArray gaussianSmooth(INDArray target) {
    return gaussianSmooth(target, 2.5);
  }

  protected static INDArray gaussianSmooth(INDArray target, double sd) {
    INDArray matrix = target.dup('c');
    INDArray kernel = Nd4j.zeros(matrix.rows(), 1, 'c');
    INDArray smooth = Nd4j.zerosLike(target);
    // convolution
    for(int t=0; t<kernel.length(); t++) {
      NormalDistribution dist = new NormalDistribution(t, sd);
      for(int k=0; k<kernel.length(); k++) {
        kernel.putScalar(k, dist.density(k));
      }
      INDArray conv = matrix.mulColumnVector(kernel); // TODO: mul takes a long time
      smooth.getRow(t).assign(conv.sum(0)); // TODO: sum takes a long time
    }
    return smooth;
  }
  
  /**
   * Returns a matrix [Tx1] that contains cosine distances between forward and backward layer.
   */
  protected static INDArray deviation(INDArray fw, INDArray bw) {
    INDArray dev = Nd4j.zeros(fw.rows(), 1);
    for(int t = 1; t < dev.rows(); t++) { // calculate first derivative in cosine distance
      double fwd1 = (t < dev.rows() - 1) ? // FW is too late
          Transforms.cosineDistance(fw.getRow(t), fw.getRow(t+1)) : 0; 
      double bwd1 = (t > 2) ? // BW is too early
          Transforms.cosineDistance(bw.getRow(t-1), bw.getRow(t-2)) : 0; 
      //dev.putScalar(t, 0, Math.sqrt(Math.pow(fwd1, 2) + Math.pow(bwd1, 2) / 2.)); // quadratic mean
      double geom = Math.sqrt(fwd1 * bwd1);
      dev.putScalar(t, 0, Double.isNaN(geom) ? 0. : geom); // geometric mean
    }
    return dev;
  }
  
  /**
   * Returns a matrix [Tx1] that contains cosine distances between t-1 and t.
   */
  protected static INDArray deviation(INDArray target) {
    INDArray dev = Nd4j.zeros(target.rows(), 1);
    for(int t = 1; t < dev.rows(); t++) {
      dev.putScalar(t, 0, Transforms.cosineDistance(target.getRow(t), target.getRow(t-1))); // first derivative
    }
    return dev;
  }
  
  /**
   * Returns a matrix [Tx1] that contains edges in deviation.
   */
  protected static INDArray detectEdges(INDArray dev) {
    if(dev == null) return null;
    INDArray result = Nd4j.zeros(dev.rows(), 1);
    for(int t = 1; t < result.rows() - 1; t++) {
      result.putScalar(t, 0, ((dev.getDouble(t - 1) < dev.getDouble(t)) && (dev.getDouble(t + 1) < dev.getDouble(t))) ? 1 : 0);
    }
    // overwrite first timestep values
    result.putScalar(0, 0, 1);
    return result;
  }
  
  /**
   * Returns a matrix [Tx1] that contains edges with given count in deviation.
   */
  protected static INDArray detectEdges(INDArray dev, int count) {
    if(dev == null) return null;
    INDArray peaks = Nd4j.zeros(dev.rows(), 1);
    for(int t = 1; t < peaks.rows() - 1; t++) {
      if((dev.getDouble(t - 1) < dev.getDouble(t)) && (dev.getDouble(t + 1) < dev.getDouble(t))) {
        peaks.putScalar(t, 0, dev.getDouble(t));
      } else {
        peaks.putScalar(t, 0, 0);
      }
    }
    
    INDArray result = Nd4j.zeros(dev.rows(), 1);
    
    // sort magnitudes and peaks
    INDArray[] p = Nd4j.sortWithIndices(Nd4j.toFlattened(peaks).dup(), 1, false); // index,value
    INDArray sortedPeaks = p[0]; // ranked indexes
    INDArray[] m = Nd4j.sortWithIndices(Nd4j.toFlattened(dev).dup(), 1, false); // index,value
    INDArray sortedMags = m[0]; // ranked indexes
    
    // pick N - 1 highest peaks
    for(int i = 0; i < count - 1; i++) {
      int idx = sortedPeaks.getInt(i);
      if(idx == 0) continue; // first one is always a new section
      if(peaks.getDouble(idx) == 0.) break; // no more peaks found
      result.putScalar(idx, 0, 1);
    }
    
    // fill with highest magnitudes
    int i = 0;
    while(i < dev.rows() && result.sumNumber().intValue() < count - 1) {
      int idx = sortedMags.getInt(i++);
      if(idx == 0) continue; // first one is always a new section
      if(result.getDouble(idx) == 1.) continue; // was already found as peak
      result.putScalar(idx, 0, 1);
    }
    
    // overwrite first timestep values
    result.putScalar(0, 0, 1);
    return result;
  }
  
  /**
   * Returns a matrix [Tx1] that contains cosine distances between time steps.
   */
  protected static INDArray deltaMatrix(INDArray data) {
    INDArray result = Nd4j.zeros(data.rows(), 1);
    INDArray prev = Nd4j.zeros(data.columns());
    for(int t = 0; t < data.rows(); t++) {
      INDArray vec = data.getRow(t);
      result.putScalar(t, 0, Transforms.cosineDistance(prev, vec));
      prev = vec.dup();
    }
    // overwrite first timestep values with max (might be NaN or too high)
    result.putScalar(0, 0, 1);
    return result;
  }

  /**
   * Builder pattern for creating new SECTOR Annotators.
   */
  public static class Builder {
    
    SectorAnnotator ann;
    SectorTagger tagger;
    
    protected Encoder[] encoders = new Encoder[0];
    protected ILossFunction lossFunc = LossFunctions.LossFunction.MCXENT.getILossFunction();
    protected Activation activation = Activation.SOFTMAX;
    protected boolean requireSubsampling = false;
    
    private int examplesPerEpoch = -1;
    private int maxTimeSeriesLength = -1;
    private int ffwLayerSize = 0;
    private int lstmLayerSize = 256;
    private int embeddingLayerSize = 128;
    private double learningRate = 0.01;
    private double dropOut = 0.5;
    private int iterations = 1;
    private int batchSize = 16; // number of Examples until Sample/Test
    private int numEpochs = 1;
    
    private boolean enabletrainingUI = false;
    
    public Builder() {
      tagger = new SectorTagger();
      ann = new SectorAnnotator(tagger);
    }
    
    public Builder withId(String id) {
      this.tagger.setId(id);
      return this;
    }
    
    public Builder withDataset(String datasetName, WordHelpers.Language lang) {
      ann.getProvenance().setDataset(datasetName);
      ann.getProvenance().setLanguage(lang.toString().toLowerCase());
      return this;
    }
    
    public Builder withLossFunction(LossFunctions.LossFunction lossFunc, Activation activation, boolean requireSubsampling) {
      this.lossFunc = lossFunc.getILossFunction();
      this.requireSubsampling = requireSubsampling;
      this.activation = activation;
      return this;
    }
    
    public Builder withLossFunction(ILossFunction lossFunc, Activation activation, boolean requireSubsampling) {
      this.lossFunc = lossFunc;
      this.requireSubsampling = requireSubsampling;
      this.activation = activation;
      return this;
    }
    
    public Builder withModelParams(int ffwLayerSize, int lstmLayerSize, int embeddingLayerSize) {
      this.ffwLayerSize = ffwLayerSize;
      this.lstmLayerSize = lstmLayerSize;
      this.embeddingLayerSize = embeddingLayerSize;
      return this;
    }
        
    public Builder withTrainingParams(double learningRate, double dropOut, int examplesPerEpoch, int batchSize, int numEpochs) {
      this.learningRate = learningRate;
      this.dropOut = dropOut;
      this.examplesPerEpoch = examplesPerEpoch;
      this.batchSize = batchSize;
      this.numEpochs = numEpochs;
      return this;
    }
    
    public Builder withTrainingParams(double learningRate, double dropOut, int examplesPerEpoch, int maxTimeSeriesLength, int batchSize, int numEpochs) {
      this.learningRate = learningRate;
      this.dropOut = dropOut;
      this.examplesPerEpoch = examplesPerEpoch;
      this.batchSize = batchSize;
      this.maxTimeSeriesLength = maxTimeSeriesLength;
      this.numEpochs = numEpochs;
      return this;
    }
        
    public Builder withInputEncoders(String desc, Encoder bagEncoder, Encoder embEncoder, Encoder flagEncoder) {
      tagger.setInputEncoders(bagEncoder, embEncoder, flagEncoder);
      ann.getProvenance().setFeatures(desc);
      ann.addComponent(bagEncoder);
      ann.addComponent(embEncoder);
      ann.addComponent(flagEncoder);
      return this;
    }
    
    public Builder withTargetEncoder(Encoder targetEncoder) {
      tagger.setTargetEncoder(targetEncoder);
      ann.addComponent(targetEncoder);
      return this;
    }
    
    public Builder withExistingComponents(SectorAnnotator parent) {
      for(Map.Entry<String, AnnotatorComponent> comp : parent.components.entrySet()) {
        //if(!ann.components.containsKey(comp.getKey())) ann.addComponent(comp.getValue()); // String Key match
        if(!ann.components.containsValue(comp.getValue())) ann.addComponent(comp.getValue()); // Instance match
      }
      return this;
    }
        
    public Builder enableTrainingUI(boolean enable) {
      this.enabletrainingUI = enable;
      return this;
    }
    
    /** pretrain encoders */
    public Builder pretrain(Dataset train) {
      for(Encoder e : encoders) {
        e.trainModel(train.streamDocuments());
      }
      return this;
    }
    
    public SectorAnnotator build() {
      tagger.buildSECTORModel(ffwLayerSize, lstmLayerSize, embeddingLayerSize, iterations, learningRate, dropOut, lossFunc, activation);
      if(enabletrainingUI) tagger.enableTrainingUI();
      tagger.setRequireSubsampling(requireSubsampling);
      tagger.setTrainingParams(examplesPerEpoch, maxTimeSeriesLength, batchSize, numEpochs, true);
      ann.getProvenance().setTask(tagger.getId());
      tagger.setName(ann.getProvenance().toString());
      tagger.appendTrainLog(printParams());
      return ann;
    }
    
    private String printParams() {
      StringBuilder line = new StringBuilder();
      line.append("TRAINING PARAMS: ").append(tagger.getName()).append("\n");
      line.append("\nInput Encoders:\n");
      for(Encoder e : tagger.getEncoders()) {
        line.append(e.getId()).append("\t").append(e.getClass().getSimpleName()).append("\t").append(e.getEmbeddingVectorSize()).append("\n");
      }
      line.append("\nTarget Encoders:\n");
      for(Encoder e : tagger.getTargetEncoders()) {
        line.append(e.getId()).append("\t").append(e.getClass().getSimpleName()).append("\t").append(e.getEmbeddingVectorSize()).append("\n");
      }
      line.append("\nNetwork Params:\n");
      line.append("FF").append("\t").append(ffwLayerSize).append("\n");
      line.append("BLSTM").append("\t").append(lstmLayerSize).append("\n");
      line.append("EMB").append("\t").append(embeddingLayerSize).append("\n");
      line.append("\nTraining Params:\n");
      line.append("examples per epoch").append("\t").append(examplesPerEpoch).append("\n");
      line.append("max time series length").append("\t").append(maxTimeSeriesLength).append("\n");
      line.append("epochs").append("\t").append(numEpochs).append("\n");
      line.append("iterations").append("\t").append(iterations).append("\n");
      line.append("batch size").append("\t").append(batchSize).append("\n");
      line.append("learning rate").append("\t").append(learningRate).append("\n");
      line.append("dropout").append("\t").append(dropOut).append("\n");
      line.append("loss").append("\t").append(lossFunc.toString()).append(requireSubsampling ? " (1-hot subsampled)" : " (1-hot/n-hot)").append("\n");
      line.append("\n");
      return line.toString();
    }

  }
  
}
