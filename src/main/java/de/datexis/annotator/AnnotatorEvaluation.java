package de.datexis.annotator;

import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import java.util.Collection;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Superclass for Evaluation (will replace ModelAnnotation)
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public abstract class AnnotatorEvaluation {

  protected Logger log = LoggerFactory.getLogger(AnnotatorEvaluation.class);

  public static enum Measure {TP, FP, TN, FN};
  
  protected String experimentName;
  protected Annotation.Source expectedSource, predictedSource;
  protected int countAnnotations, countExamples, countDocs, countSentences, countTokens;
  
  /**
   * Evaluates an Annotator based on the implemented scoring functions.
   * @param experimentName - name of the Experiment
   * @param expected - expected Annotation Source, e.g. Annotation.Source.GOLD
   * @param predicted - predicted Annotation Source, e.g. Annotation.Source.PRED
   */
  public AnnotatorEvaluation(String experimentName, Annotation.Source expected, Annotation.Source predicted) {
    this.experimentName = experimentName;
    this.expectedSource = expected;
    this.predictedSource = predicted;
  }
  
  public AnnotatorEvaluation(String experimentName) {
    this(experimentName, Annotation.Source.GOLD, Annotation.Source.PRED);
  }
  
  /**
   * Calculate all scores for a given Dataset.
   */
  public void calculateScores(Dataset dataset) {
    calculateScores(dataset.getDocuments());
  }
  
  /**
   * Calculate all scores for a list of Documents.
   */
  public abstract void calculateScores(Collection<Document> docs);
  
  /**
   * Calculate all scores for a given Dataset an Annoation level.
   * This needs to be called before you can access getScore(). Existing scores will be cleared.
   * @param documents - the test documents to evaluate
   * @param annotationClass - the Annotation type that is being evaluated, requires a matching function
   */
  //public abstract void calculateScoresFromAnnotations(Iterable<Document> documents, Class<? extends Annotation> annotationClass);

  /**
   * Calculate all scores for a given Dataset an Annoation level.
   * This needs to be called before you can access getScore(). Existing scores will be cleared.
   * @param documents - the test documents to evaluate
   * @param spanClass - the Span type that is being evaluated, e.g. Token.class or Sentence.class
   * @param tagClass - the Tag type that is attached to the spans, e.g. BIOES.class
   */
  //public abstract void calculateScoresFromTags(Iterable<Document> documents, Class<? extends Span> spanClass, Class<? extends Tag> tagClass);
  
  /**
   * @return the primary score from last call to calculateScores()
   */
  public abstract double getScore();
  
  /**
   * @return the number of documents used for evaluation
   */
  public double countDocuments() {
    return countDocs;
  }

  /**
   * @return the number of sentences seen at evaluation
   */
  public double countSentences() {
    return countSentences;
  }
  
  /**
   * @return the number of tokens seen at evaluation
   */
  public double countTokens() {
    return countTokens;
  }
  
  
  /**
   * @return the number of examples (e.g. Annotations or Sentences) used at evaluation
   */
  public double countExamples() {
    return countExamples;
  }
  
  
  /**
   * @return the number of GOLD Annotations expected for evaluation
   */
  public double countAnnotations() {
    return countAnnotations;
  }
  
  
  /**
   * @return format Double for Table with 2 decimals
   */
  protected static String fDbl(double d) {
    return String.format(Locale.ROOT, "%6.2f", d * 100);
  }
  
  /**
   * @return format Integer for Table
   */
  protected static String fInt(double d) {
    return String.format(Locale.ROOT, "%6d", (int)d);
  }
  
  /**
   * @return format Integer for Table
   */
  protected static String fStr(String s, int spaces) {
    return String.format(Locale.ROOT, "%-" + spaces +"s", s);
  }
  
}
