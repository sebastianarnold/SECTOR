package de.datexis.sector.eval;

import de.datexis.annotator.AnnotatorEvaluation;
import de.datexis.encoder.LookupCacheEncoder;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.sector.encoder.ClassTag;
import de.datexis.sector.encoder.HeadingEncoder;
import de.datexis.sector.encoder.HeadingTag;
import de.datexis.sector.model.SectionAnnotation;
import java.util.Collection;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Combined Evaluation of Headings per Sentence, Segmentation and Classification per Segment
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class SectorEvaluation extends AnnotatorEvaluation {

  protected final static Logger log = LoggerFactory.getLogger(SectorEvaluation.class);

  protected boolean enableSentenceEval = true;
  protected boolean enableSegmentEval = true;
  protected boolean enableSegmentationEval = true;
  
  protected int countSections;
  protected int countPredictions;
  
  protected ClassificationEvaluation sentenceClassEval;
  protected ClassificationEvaluation segmentClassEval;
  protected SegmentationEvaluation segmentationEval;
  protected Class targetEncoderClass;
  
  /**
   * Initialize evaluation for classification and segmentation
   */
  public SectorEvaluation(String experimentName, Annotation.Source expected, Annotation.Source predicted, LookupCacheEncoder targetEncoder) {
    super(experimentName, expected, predicted);
    sentenceClassEval = new ClassificationEvaluation(experimentName, expectedSource, predictedSource,  targetEncoder, 3);
    segmentClassEval = new ClassificationEvaluation(experimentName, expectedSource, predictedSource, targetEncoder, 3);
    segmentationEval = new SegmentationEvaluation(experimentName);
    this.targetEncoderClass = targetEncoder.getClass();
  }
  
  /**
   * Initialize evaluation for segmentation only
   */
  public SectorEvaluation(String experimentName, Annotation.Source expected, Annotation.Source predicted) {
    super(experimentName, expected, predicted);
    sentenceClassEval = null;
    segmentClassEval = null;
    segmentationEval = new SegmentationEvaluation(experimentName);
  }

  /**
   * Enable/disable evaluation for sentence-based classification (Prec/Rec/F1).
   */
  public SectorEvaluation withSentenceClassEvaluation(boolean enable) {
    this.enableSentenceEval = enable;
    return this;
  }
  
  /**
   * Enable/disable evaluation for segmentation (Pk/WD).
   */
  public SectorEvaluation withSegmentationEvaluation(boolean enable) {
    this.enableSegmentationEval = enable;
    return this;
  }
  
  /**
   * Enable/disable evaluation for segment-based classification (Prec/Rec/F1).
   */
  public SectorEvaluation withSegmentClassEvaluation(boolean enable) {
    this.enableSegmentEval = enable;
    return this;
  }
  
  /**
   * Enable/disable recalculation of K for each document in the evaluation.
   * if FALSE, we use a fixed K for all documents.
   */
  public SectorEvaluation withRecalculateK(boolean enabled) {
    segmentationEval.withRecalculateK(enabled);
    return this;
  }
  
  /**
   * Enable/disable merging adjacent sections with same label into one (in both GOLD and PRED).
   */
  public SectorEvaluation withMergeEnabled(boolean enabled) {
    segmentationEval.withMergeEnabled(enabled);
    return this;
  }
  
  /**
   * Evaluate the whole SECTOR model.
   * For Multi-label evaluation of Headings:
   * - <b>requires expected and predicted HeadingTags</b> attached to every Sentence in the test set
   * - <b>requires class distribution Vectors</b> attached to the HeadingTags
   * For Segmentation evaluation:
   * - <b>requires expected and predicted SectionAnnotations</b> attached to Documents
   * For Classification evaluation:
   * - <b>requires class distribution Vectors</b> attached to expected and predicted Annotations
   */
  @Override
  public void calculateScores(Collection<Document> docs) {
    
    log.info("Evaluating SECTOR...");
    
    countSections = 0;
    countPredictions = 0;
    countExamples = 0;
    countDocs = 0;
    if(enableSentenceEval && sentenceClassEval != null) {
      log.info("calculating sentence scores from tags...");
      if(targetEncoderClass == HeadingEncoder.class) {
        sentenceClassEval.calculateScoresFromTags(docs, Sentence.class, HeadingTag.class);
      } else { // e.g. ClassEncoder or ParVecEncoder
        sentenceClassEval.calculateScoresFromTags(docs, Sentence.class, ClassTag.class);
      }
    }
    if(enableSegmentEval && segmentClassEval != null) {
      log.info("calculating segment scores from annotations...");
      segmentClassEval.calculateScoresFromAnnotations(docs, SectionAnnotation.class, true);
    }
    if(enableSegmentationEval && segmentationEval != null) {
      log.info("calculating segmentation scores from annotations...");
      segmentationEval.calculateScoresFromAnnotations(docs, SectionAnnotation.class);
    }
    log.info("done.");
    
    for(Document doc : docs) {
      Collection<SectionAnnotation> expected = doc.getAnnotations(expectedSource, SectionAnnotation.class);
      Collection<SectionAnnotation> predicted = doc.getAnnotations(predictedSource, SectionAnnotation.class);
      countDocs++;
      countExamples += doc.countSentences();
      countSections += expected.size();
      countPredictions += predicted.size();
    }
  }

  /**
   * Return the MAP score from this Evaluation run.
   */
  @Override
  public double getScore() {
    if(enableSegmentEval && segmentClassEval != null) return segmentClassEval.getScore();
    if(enableSentenceEval && sentenceClassEval != null) return sentenceClassEval.getScore();
    return 0;
  }

  /**
   * Print out some stats about the Dataset.
   */
  public static String printDatasetStats(Dataset dataset) {
    StringBuilder line = new StringBuilder();
    line.append("DATASET:\t").append(dataset.getName()).append("\n");
    line.append("#Docs\t#Sents\t#Tokens\t#Anns\n");
    line.append(String.format(Locale.ROOT, "%,d",dataset.countDocuments())).append("\t");
    line.append(String.format(Locale.ROOT, "%,d",dataset.countSentences())).append("\t");
    line.append(String.format(Locale.ROOT, "%,d",dataset.countTokens())).append("\t");
    line.append(String.format(Locale.ROOT, "%,d",dataset.countAnnotations(Annotation.Source.GOLD))).append("\t");
    //line.append(String.format(Locale.ROOT, "%,d",trainCount)).append("\t");
    //line.append(Timer.millisToLongDHMS(trainTime)).append("\n");
    line.append("------------------------------------------------------------------------------\n");
    System.out.println(line.toString());
    return line.toString();
  }

  /**
   * Print out the Evaluation results table.
   */
  public String printEvaluationStats() {

    if(sentenceClassEval == null) enableSentenceEval = false;
    if(segmentationEval == null) enableSegmentationEval = false;
    if(segmentClassEval == null) enableSegmentEval = false;

    StringBuilder line = new StringBuilder();
    line.append("SECTOR EVALUATION [micro-avg] ").append(targetEncoderClass.getSimpleName()).append("\n")
        .append("|statistics ---\t");
    if(enableSentenceEval)     line.append("|sentence classification -------------------------------------\t");
    if(enableSegmentationEval) line.append("|segmentation --------------------------------\t");
    if(enableSegmentEval)      line.append("|segment classification ----------------------------");
    line.append("\n")
        .append("||docs|\t|sents|\t");
    if(enableSentenceEval)     line.append("| A@1\t A@3\t P@1\t P@3\t R@1\t R@3\t MAP\t");
    if(enableSegmentationEval) line.append("| |exp|\t |relv|\t |pred|\t |retr|\t Pk\t WD\t");
    if(enableSegmentEval)      line.append("| A@1\t A@3\t P@1\t P@3\t R@1\t R@3\t MAP");
    line.append("\n");

    // statistics
    line.append(fInt(this.countDocuments())).append("\t");
    line.append(fInt(this.countExamples())).append("\t");
    
    // Topic Classification: label(s) per sentence
    if(enableSentenceEval) {
      line.append(fDbl(sentenceClassEval.getAccuracy())).append("\t");
      line.append(fDbl(sentenceClassEval.getAccuracyK())).append("\t");
      line.append(fDbl(sentenceClassEval.getPrecision1())).append("\t");
      line.append(fDbl(sentenceClassEval.getPrecisionK())).append("\t");
      line.append(fDbl(sentenceClassEval.getRecall1())).append("\t");
      line.append(fDbl(sentenceClassEval.getRecallK())).append("\t");
      line.append(fDbl(sentenceClassEval.getMAP())).append("\t");
    }
    
    // Topic Segementation: segmentation
    if(enableSegmentationEval) {
      line.append(fInt(this.countSections())).append("\t");
      line.append(fInt(segmentationEval.getCountExpected())).append("\t");
      line.append(fInt(this.countPredictions())).append("\t");
      line.append(fInt(segmentationEval.getCountPredicted())).append("\t");
      line.append(fDbl(segmentationEval.getPk())).append("\t");
      line.append(fDbl(segmentationEval.getWD())).append("\t");
    }
    
    // Topic Classification: label(s) per segment
    if(enableSegmentEval) {
      line.append(fDbl(segmentClassEval.getAccuracy())).append("\t");
      line.append(fDbl(segmentClassEval.getAccuracyK())).append("\t");
      line.append(fDbl(segmentClassEval.getPrecision1())).append("\t");
      line.append(fDbl(segmentClassEval.getPrecisionK())).append("\t");
      line.append(fDbl(segmentClassEval.getRecall1())).append("\t");
      line.append(fDbl(segmentClassEval.getRecallK())).append("\t");
      line.append(fDbl(segmentClassEval.getMAP())).append("\t");
    }
    
    line.append("\n");
    System.out.println(line.toString());
    return line.toString();
  }

  /**
   * Print out Evaluation results table for single classes.
   */
  public String printSingleClassStats() {
    if(enableSegmentEval && segmentClassEval.numClasses < 50) {
      return printSingleClassStats(segmentClassEval);
    } else if(enableSentenceEval && sentenceClassEval.numClasses < 50) {
      return printSingleClassStats(sentenceClassEval);
    } else {
      return "Too many classes for single-class stats";
    }
  }
  
  public static String printSingleClassStats(ClassificationEvaluation eval) {
    
    StringBuilder line = new StringBuilder();    
    line.append("SINGLE-LABEL CLASSIFICATION [performance per class]\n")
        .append("No\tClass\t#Examples\tTP\tFP\tAcc\tPrec\tRec\tF1\n");
    for(int c = 0; c < eval.numClasses; ++c) {
      line.append(c).append("\t");
      line.append(eval.eval.getClassLabel(c)).append("\t");
      line.append(fInt(eval.eval.getConfusionMatrix().getActualTotal(c))).append("\t");
      line.append(fInt(eval.eval.getTruePositives().getCount(c))).append("\t");
      line.append(fInt(eval.eval.getFalsePositives().getCount(c))).append("\t");
      line.append(fDbl(eval.getAccuracy(c))).append("\t"); // Accuracy = Recall
      line.append(fDbl(eval.getPrecision(c))).append("\t");
      line.append(fDbl(eval.getRecall(c))).append("\t");
      line.append(fDbl(eval.getF1(c))).append("\t");
      line.append("\n");
    }
    /*line.append("TOTAL [macro-avg]\t\t");
    line.append(fInt(eval.countExamples())).append("\t");
    line.append(fInt(eval.eval.getTruePositives().totalCount())).append("\t");
    line.append(fInt(eval.eval.getFalsePositives().totalCount())).append("\t");
    line.append(fDbl(eval.getAccuracy())).append("\t"); // Accuracy = Micro F1
    line.append(fDbl(eval.getMacroPrecision())).append("\t");
    line.append(fDbl(eval.getMacroRecall())).append("\t");
    line.append(fDbl(eval.getMacroF1())).append("\t");
    line.append("\n");*/
    System.out.println(line.toString());
    return line.toString();
  }

  private double countSections() {
    return countSections;
  }

  private double countPredictions() {
    return countPredictions;
  }

}
