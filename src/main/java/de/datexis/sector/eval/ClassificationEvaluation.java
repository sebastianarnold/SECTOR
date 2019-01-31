package de.datexis.sector.eval;

import de.datexis.annotator.AnnotatorEvaluation;
import de.datexis.encoder.LookupCacheEncoder;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.model.Span;
import de.datexis.model.tag.Tag;
import java.io.Serializable;
import java.util.Collection;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.EvaluationAveraging;
import org.nd4j.evaluation.EvaluationUtils;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.LoggerFactory;

/**
 * Evaluates Precision/Recall/F1 for Sentence-based class labeling (e.g. Sentence Classification).
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class ClassificationEvaluation extends AnnotatorEvaluation implements IEvaluation<ClassificationEvaluation> {

  protected LookupCacheEncoder encoder;
  protected int numClasses;
  protected int K;
  protected Evaluation eval;

  /** average precision */
  protected double mrrsum = 0., mapsum = 0., p1sum = 0., r1sum = 0., pksum = 0., rksum = 0.;
  
  public ClassificationEvaluation(String experimentName, LookupCacheEncoder encoder) {
    this(experimentName, Annotation.Source.GOLD, Annotation.Source.PRED, encoder, 3);
  }
  
  public ClassificationEvaluation(String experimentName, Annotation.Source expected, Annotation.Source predicted, LookupCacheEncoder encoder, int K) {
    super(experimentName, expected, predicted);
    this.K = K;
    this.encoder = encoder;
    this.numClasses = (int) encoder.getEmbeddingVectorSize();
    log = LoggerFactory.getLogger(ClassificationEvaluation.class);
    clear();
  }

  protected void clear() {
    eval = new Evaluation(encoder.getWords(), K);
    countDocs = 0;
    countExamples = 0;
    mrrsum = 0.;
    mapsum = 0.;
    p1sum = 0.;
    r1sum = 0.;
    pksum = 0.;
    rksum = 0.;
  }
  
  /*protected double getCount(Measure m, int classIdx) {
    return (double) counts.get(m).getCount(classIdx);
  }*/
  
  @Override
  public double getScore() {
    return getMAP();
  }
  
  /**
   * Not used. Please use calculateScoresFromAnnotations or calculateScoresFromTags.
   */
  @Override
  public void calculateScores(Collection<Document> docs) {
    throw new UnsupportedOperationException("Not implemented yet.");
  }
  
  /**
   * Calculates Evaluation from Annotations in the Documents.
   * - <b>requires expected and predicted Annotations</b> attached to Documents (will match expected to predicted via position)
   * - <b>requires class distribution Vectors</b> attached to expected and predicted Annotations
   * @param matchAllPredicted - if TRUE, all remaining unmatched predicted annotations will be matched to expected via position, otherwise they are ignored
   */
  public void calculateScoresFromAnnotations(Collection<Document> documents, Class<? extends Annotation> annotationClass, boolean matchAllPredicted) {
    Map<Annotation, Boolean> matched = new IdentityHashMap<>();
    countDocs += documents.size();
    for(Document doc : documents) {
      // match relevant annotations to predicted annotations
      for(Annotation expected : doc.getAnnotations(expectedSource, annotationClass)) {
        Optional<? extends Annotation> predicted = doc.getAnnotationMaxOverlap(predictedSource, annotationClass, expected);
        if(predicted.isPresent()) {
          matched.put(predicted.get(), true);
          INDArray r = expected.getVector(encoder.getClass()).transpose();
          INDArray p = predicted.get().getVector(encoder.getClass()).transpose();
          evalExample(r, p);
        } else {
          log.warn("Could not match predicted Annotation for expected Annotation {}-{}", expected.getBegin(), expected.getEnd());
        }
      }
      if(!matchAllPredicted) continue;
      // match additional predicted annotations to expected
      for(Annotation predicted : doc.getAnnotations(predictedSource, annotationClass)) {
        if(!matched.containsKey(predicted)) {
          Optional<? extends Annotation> expected = doc.getAnnotationMaxOverlap(expectedSource, annotationClass, predicted);
          if(expected.isPresent()) {
            INDArray r = expected.get().getVector(encoder.getClass()).transpose();
            INDArray p = predicted.getVector(encoder.getClass()).transpose();
            evalExample(r, p);
          } 
        }
      }
    }
  }
  
  /**
   * Calculates Evaluation from Tags in the Spans.
   * - <b>requires expected and predicted Tags</b> attached to the given Span class
   * - <b>requires class distribution Vectors</b> attached to the Tags
   */
  public <T extends Tag> void calculateScoresFromTags(Collection<Document> documents, Class<? extends Span> spanClass, Class<T> tagClass) {
    countDocs += documents.size();
    for(Document doc : documents) {
      for(Span s : doc.getStream(spanClass).collect(Collectors.toList())) {
        // use encoder to ensure unknown class
        INDArray r = s.getTag(expectedSource, tagClass).getVector().transpose();
        INDArray p = s.getTag(predictedSource, tagClass).getVector().transpose();
        evalExample(r, p);
      }
    }
  }
  
  /**
   * Update scores from a single Example prediction
   * @param Y - correct labels e {0,1}^d
   * @param Z - predicted labels e R^d@param Y
   */
  public void evalExample(INDArray Y, INDArray Z) {
    // pre-calculate ranked indices
    INDArray[] z = Nd4j.sortWithIndices(Nd4j.toFlattened(Z).dup(), 1, false); // index,value
    if(z[0].sumNumber().doubleValue() == 0.)
      log.warn("Sort on zero vector - please check vector dimensions!");
    INDArray Zi = z[0]; // ranked indexes
    eval.eval(Y, Z);
    mapsum += AP(Y, Z, Zi);
    mrrsum += RR(Y, Z, Zi);
    p1sum += Prec(Y, Z, Zi, 1);
    r1sum += Rec(Y, Z, Zi, 1);
    pksum += Prec(Y, Z, Zi, K);
    rksum += Rec(Y, Z, Zi, K);
    countExamples++;
  }
  
  /** safe division, where n/0 = 0 */
  protected double div(double n, double d) {
    if(d == 0.0) return 0.0;
    else return n / d;
  }
  
  /**
   * get position of index idx in ranked labels l
   * @return position between 1 and length(l)
   */
  protected static int rank(int idx, INDArray l) {
    for(int i = 0; i < l.length(); ++i) {
      if(l.getInt(i) == idx) return i + 1;
    }
    throw new IllegalArgumentException("index does not exist in labels");
  }
  
  /**
   * Reciprocal Rank.
   * https://en.wikipedia.org/wiki/Mean_reciprocal_rank
   * @param Y - correct labels e {0,1}^d
   * @param Z - predicted labels e R^d
   * @return 
   */
  private double RR(INDArray Y, INDArray Z, INDArray Zi) {
    int ri = maxIndex(Y); // relevant index
    if(ri >= 0) {
      double r = rank(ri, Zi);
      return 1. / (double) r;
    } else { // there is no relevant label
      return 0.; 
    }
  }
  
  /**
   * Standard Average Precision.
   * https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
   * @param Y - correct labels e {0,1}^d
   * @param Z - predicted labels e R^d
   * @param Zi - ranked indices of predicted labales
   * @return 
   */
  private double AP(INDArray Y, INDArray Z, INDArray Zi) {
    // sum over all labels
    double sum = 0;
    int count = 0;
    for(int k = 0; k < Y.length(); k++) {
      int idx = Zi.getInt(k);
      if(Y.getDouble(idx) > 0.) { // check if kth prediction is relevant
        sum += Prec(Y, Z, Zi, k + 1);
        count++;
      }
    }
    assert(count == 1);
    if(count > 0) return sum / (double) count;
    else return 0;
  }
  
  /**
   * Precision at K. Proportion of top-K documents that are relevant.
   * @param Y - correct labels e {0,1}^d
   * @param Z - predicted labels e R^d
   * @param Zi - ranked indices of predicted labales
   */
  private double Prec(INDArray Y, INDArray Z, INDArray Zi, int k) {
    double sum = 0;
    for(int i = 0; i < k; i++) {
      int idx = Zi.getInt(i); // index of top-i prediction
      if(Y.getDouble(idx) > 0.) sum++;
    }
    return sum / (double) k;
  }
  
  /**
   * Recall at K. Proportion of relevant documents that are in top-K.
   * @param Y - correct labels e {0,1}^d
   * @param Z - predicted labels e R^d
   * @param Zi - ranked indices of predicted labales
   */
    private double Rec(INDArray Y, INDArray Z, INDArray Zi, int k) {
    if(Y.sumNumber().doubleValue() == 0) return 0.; // there is no relevant label
    double sum = 0;
    for(int i = 0; i < k; i++) {
      int idx = Zi.getInt(i); // index of top-i prediction
      if(Y.getDouble(idx) > 0.) sum++;
    }
    return sum / Y.sumNumber().doubleValue();
  }
  
  /**
   * @return the relevant index Yi == 1
   */
  protected static int maxIndex(INDArray Y) {
    int idx = -1;
    double max = Double.MIN_VALUE;
    for(int i=0; i < Y.length(); ++i) {
      if(Y.getDouble(i) > max) {
        max = Y.getDouble(i);
        idx = i;
      }
    }
    return idx;
  }
  
  /**
   * Micro/Macro Accuracy
   */
  public double getAccuracy() {
    return eval.accuracy();
  }
  
  public double getAccuracyK() {
    return eval.topNAccuracy();
  }
  
  /**
   * Accuracy per class
   * @param c - class index
   */
  protected double getAccuracy(int c) {
    return div(eval.truePositives().get(c), eval.positive().get(c));
  }
  
  /**
   * Micro Precision (average precision over all examples).
   * This is the CoNLL2003 Precision.
   * @return precision = correctChunk / foundGuessed
   */
  public double getMicroPrecision() {
    return eval.precision(EvaluationAveraging.Micro);
  }
  
  
  /**
   * Macro Precision (average Precision over all classes).
   */
  public double getMacroPrecision() {
    //return eval.precision(EvaluationAveraging.Macro); // will exclude classes that have no prediction!
    double sum = 0.0;
    for (int c = 0; c < numClasses; c++) {
      sum += getPrecision(c);
    }
    return sum / numClasses;
  }
  
  /**
   * Precision per class
   * @param c - class index
   */
  protected double getPrecision(int c) {
    return eval.precision(c);
  }
  
  /**
   * Micro Recall (average recall over all examples).
   * This is the CoNLL2003 Recall.
   * @return recall = correctChunk / foundCorrect
   */
  public double getMicroRecall() {
    return eval.recall(EvaluationAveraging.Micro);
  }
  
  /**
   * Macro Recall (average recall over all classes).
   */
  public double getMacroRecall() {
    //return eval.recall(EvaluationAveraging.Macro); // will exclude classes that have no prediction!
    double sum = 0.0;
    for (int c = 0; c < numClasses; c++) {
      sum += getRecall(c);
    }
    return sum / numClasses;
  }
  
  /**
   * Recall per class
   * @param c - class index
   */
  protected double getRecall(int c) {
    return eval.recall(c);
  }
    
  /**
   * Micro F1 score (average F1 over all examples).
   * This is CoNLL2003 NER-style F1
   * @return $FB1 = 2*$precision*$recall/($precision+$recall) if ($precision+$recall > 0)
   */
  public double getMicroF1() {
    return eval.f1(EvaluationAveraging.Micro);
  }
  
  /**
   * Macro F1 score (average F1 over all classes).
   */
  public double getMacroF1() {
    //return eval.f1(EvaluationAveraging.Macro); // will exclude classes that have no prediction!
    double sum = 0.0;
    for (int c = 0; c < numClasses; c++) {
      sum += getF1(c);
    }
    return sum / numClasses;
  }
  
  /**
   * F1 score per class
   * @param c - class index
   */
  protected double getF1(int c) {
    return eval.f1(c);
  }
  
  protected double getMRR() {
    return mrrsum / countExamples;
  }
  
  public double getMAP() {
    return mapsum / countExamples;
  }
  
  public double getPrecisionK() {
    return pksum / countExamples;
  }
  
  public double getRecallK() {
    return rksum / countExamples;
  }
  
  public double getPrecision1() {
    return p1sum / countExamples;
  }
  
  public double getRecall1() {
    return r1sum / countExamples;
  }
  
  @Override
  public void eval(INDArray labels, INDArray networkPredictions) {
    for(int i = 0; i < labels.rows(); i++) {
      evalExample(labels.getRow(i), networkPredictions.getRow(i));
    }
  }

  @Override
  public void eval(INDArray labels, INDArray networkPredictions, List<? extends Serializable> recordMetaData) {
    eval(labels, networkPredictions);
  }

  @Override
  public void eval(INDArray labels, INDArray networkPredictions, INDArray maskArray) {
    if(maskArray == null) {
      if(labels.rank() == 3) {
        evalTimeSeries(labels, networkPredictions, maskArray);
      } else {
        eval(labels, networkPredictions);
      }
      return;
    }
    if(labels.rank() == 3 && maskArray.rank() == 2) {
      //Per-output masking
      evalTimeSeries(labels, networkPredictions, maskArray);
      return;
    }

    throw new UnsupportedOperationException(
        this.getClass().getSimpleName() + " does not support per-output masking");
  }

  @Override
  public void evalTimeSeries(INDArray labels, INDArray predicted) {
    evalTimeSeries(labels, predicted, null);
  }

  @Override
  public void evalTimeSeries(INDArray labels, INDArray predictions, INDArray labelsMask) {
    Pair<INDArray, INDArray> pair = EvaluationUtils.extractNonMaskedTimeSteps(labels, predictions, labelsMask);
    if(pair == null){
        //No non-masked steps
        return;
    }
    INDArray labels2d = pair.getFirst();
    INDArray predicted2d = pair.getSecond();

    eval(labels2d, predicted2d);
  }

  @Override
  public void merge(ClassificationEvaluation other) {
    throw new UnsupportedOperationException("Not implemented yet.");
  }

  @Override
  public void reset() {
    throw new UnsupportedOperationException("Not implemented yet.");
  }

  @Override
  public String stats() {
    throw new UnsupportedOperationException("Not implemented yet.");
  }

  @Override
  public String toJson() {
    throw new UnsupportedOperationException("Not implemented yet.");
  }

  @Override
  public String toYaml() {
    throw new UnsupportedOperationException("Not implemented yet.");
  }
  
  /**
   * Print micro-averaged scores for evaluation @K
   * @return 
   */
  public String printClassificationAtKStats() {
    ClassificationEvaluation eval = this;
    StringBuilder line = new StringBuilder();
    line.append(" Acc@1\t Acc@").append(K).append("\t P@1\t P@").append(K).append("\t R@1\t R@").append(K).append("\t MAP\n");
    line.append(fDbl(eval.getAccuracy())).append("\t");
    line.append(fDbl(eval.getAccuracyK())).append("\t");
    line.append(fDbl(eval.getPrecision1())).append("\t");
    line.append(fDbl(eval.getPrecisionK())).append("\t");
    line.append(fDbl(eval.getRecall1())).append("\t");
    line.append(fDbl(eval.getRecallK())).append("\t");
    line.append(fDbl(eval.getMAP())).append("\t");
    line.append("\n");
    //System.out.println(line.toString());
    return line.toString();
  }
  
  public String printClassificationStats() {
    ClassificationEvaluation eval = this;
    StringBuilder line = new StringBuilder();    
    line.append(" count\t TP\t FP\t MRR\t P@1\t MAP\t mPrec\t mRec\t mF1\n");
    line.append(fInt(eval.countExamples())).append("\t");
    line.append(fInt(eval.eval.getTruePositives().totalCount())).append("\t");
    line.append(fInt(eval.eval.getFalsePositives().totalCount())).append("\t");
    line.append(fDbl(eval.getMRR() / 100.)).append("\t");
    line.append(fDbl(eval.getAccuracy())).append("\t"); // Accuracy = Micro F1
    line.append(fDbl(eval.getMAP())).append("\t");
    line.append(fDbl(eval.getMacroPrecision())).append("\t");
    line.append(fDbl(eval.getMacroRecall())).append("\t");
    line.append(fDbl(eval.getMacroF1())).append("\t");
    line.append("\n");
    //System.out.println(line.toString());
    return line.toString();
  }
  
}
