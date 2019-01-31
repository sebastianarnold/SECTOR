package de.datexis.sector.eval;

import de.datexis.annotator.AnnotatorEvaluation;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.sector.model.SectionAnnotation;
import java.util.ArrayList;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

public class SegmentationEvaluation extends AnnotatorEvaluation {

  protected final double DEFAULT_SCORE = 0D;
  
  protected double countExp = 0., countPred = 0., pksum = 0., wdsum = 0.;
  
  /** whether to recalculate the value of K for each document in the evaluation */
  protected boolean enableKPerDocument = false;
  
  /** whether to merge adjacent sections with same label into one (in both GOLD and PRED) */
  protected boolean enableMergeSections = true;
  
  public SegmentationEvaluation(String experimentName) {
    this(experimentName, Annotation.Source.GOLD, Annotation.Source.PRED);
  }
  
  public SegmentationEvaluation(String experimentName, Annotation.Source expected, Annotation.Source predicted) {
    super(experimentName, expected, predicted);
    log = LoggerFactory.getLogger(SegmentationEvaluation.class);
    clear();
  }
  
  /**
   * Enable/disable recalculation of K for each document in the evaluation.
   * if FALSE, we use a fixed K for all documents.
   */
  public SegmentationEvaluation withRecalculateK(boolean enabled) {
    this.enableKPerDocument = enabled;
    return this;
  }
  
  /**
   * Enable/disable merging adjacent sections with same label into one (in both GOLD and PRED).
   */
  public SegmentationEvaluation withMergeEnabled(boolean enabled) {
    this.enableMergeSections = enabled;
    return this;
  }
  
  protected void clear() {
    countDocs = 0;
    countExamples = 0;
    pksum = 0.;
    wdsum = 0.;
    countExp = 0.;
    countPred = 0.;
  }
  
  @Override
  public double getScore() {
    return getWD();
  }
  
  @Override
  public void calculateScores(Collection<Document> docs) {
    calculateScoresFromAnnotations(docs, SectionAnnotation.class);
  }
  
  public void calculateScoresFromAnnotations(Collection<Document> docs, Class<? extends Annotation> annotationClass) {
    countDocs += docs.size();
    int k = calculateK(docs); // global K
    for(Document doc : docs) {
      if(enableKPerDocument) k = calculateK(doc); // update k per individual example
      wdsum += calculateWD(doc, k);
      pksum += calculatePk(doc, k);
      countExp += getMassesArray(doc, expectedSource).length;
      countPred += getMassesArray(doc, predictedSource).length;
    }
  }
  
  public double getWD() {
    return wdsum / countDocs;
  }
  
  public double getPk() {
    return pksum / countDocs;
  }
  
  public double getCountExpected() {
    return countExp;
  }
  
  public double getCountPredicted() {
    return countPred;
  }
  
  /**
   * Calculate Pk metric.
   * Adapted from https://github.com/cfournie/segmentation.evaluation
   */
  public double calculatePk(Document doc, int k) {
    int[] reference = getPositionsArray(doc, expectedSource);
    int[] hypothesis = getPositionsArray(doc, predictedSource);
    double sum = 0;
    double count = 0;
    for(int t = 0; t < reference.length - k; t++) {
      // calculate disagreement in window of size k
      boolean agreeRef = reference[t] == reference[t + k];
      boolean agreeHyp = hypothesis[t] == hypothesis[t + k];
      if(agreeRef != agreeHyp) sum++;
      count++;
    }
    // for some reason this case is not checked in window...?
    if(reference.length == 2 ) {
      assert(count == 0);
      boolean agreeRef = reference[0] == reference[1];
      boolean agreeHyp = hypothesis[0] == hypothesis[1];
      if(agreeRef == agreeHyp) return 0.;
      else return 1.;
    }
    if(reference.length == 1 ) return 0.;
    if(count > 0) return sum / count;
    else return 0;
  }
  
  /**
   * Calculate WD metric.
   * Adapted from https://github.com/cfournie/segmentation.evaluation
   */
  public double calculateWD(Document doc, int k) {
    int[] reference = getPositionsArray(doc, expectedSource);
    int[] hypothesis = getPositionsArray(doc, predictedSource);
    double sum = 0;
    double count = 0;
    // calculate disagreement in length - k windows
    for(int t = 0; t < reference.length - k; t++) {
      int sumRef = 0;
      int sumHyp = 0;
      // check all pairs in window if they contain a boundary
      for(int j = 0; j < k; j++) {
        if(reference[t + j] == 0) {
          log.warn("document is not correctly annotated");
          return 1.;
        }
        boolean agreeRef = reference[t + j] == reference[t + j + 1];
        boolean agreeHyp = hypothesis[t + j] == hypothesis[t + j + 1];
        if(agreeRef) sumRef++;
        if(agreeHyp) sumHyp++;
      }
      // disagree if number of boundaries in window differs
      if(sumRef != sumHyp) sum++;
      count++;
    }
    // for some reason this case is not checked in window...?
    if(reference.length == 2 ) {
      assert(count == 0);
      boolean agreeRef = reference[0] == reference[1];
      boolean agreeHyp = hypothesis[0] == hypothesis[1];
      if(agreeRef == agreeHyp) return 0.;
      else return 1.;
    }
    if(reference.length == 1 ) return 0.;
    if(count > 0) return sum / count;
    else return 0.0;
  }
  
  /**
   * @return preferred window size as half the mean segment length
   */
  public int calculateK(Collection<Document> docs) {
    int k = Math.max((int) Math.round(getMeanSegmentLength(docs) / 2.), 2);
    log.trace("setting k to {}", k);
    return k;
  }
  
  public int calculateK(Document doc) {
    double sum = 0;
    int[] masses = getMassesArray(doc, expectedSource);
    for(int c : masses) sum += c;
    int k = Math.max((int) Math.round((sum / (double) masses.length) / 2.), 2);
    return k;
  }
    
  public double getMeanSegmentLength(Collection<Document> docs) {
    double sum = 0;
    double count = 0;
    for(Document doc : docs) {
      int[] masses = getMassesArray(doc, expectedSource);
      for(int c : masses) sum += c;
      count += masses.length;
    }
    return sum / count;
  }
  
  /**
   * @return a masses array from SectionAnnotations, e.g. [1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,5,5,5,5] -> [3,6,3,2,4]
   */
  public int[] getMassesArray(Document doc, Annotation.Source source) {
    ArrayList<Integer> result = new ArrayList<>();
    int[] positions = getPositionsArray(doc, source);
    int last = 0;
    int count = 0;
    for(int curr : positions) {
      if(curr != last) {
        if(count > 0) result.add(count);
        last = curr;
        count = 0;
      }
      count++;
    }
    if(count > 0) result.add(count);
    return result.stream().mapToInt(Integer::valueOf).toArray();
  }
  
  /**
   * @return a positions array from SectionAnnotations, e.g. [1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,5,5,5,5]
   */
  public int[] getPositionsArray(Document doc, Annotation.Source source) {
    int[] array = new int[doc.countSentences()];
    int sectionIndex = 0;
    int cursor = 0;
    String currentSection, lastSection = "";
    List<SectionAnnotation> anns = doc.streamAnnotations(source, SectionAnnotation.class).sorted().collect(Collectors.toList());
    for(SectionAnnotation ann : anns) {
      int begin = doc.getSentenceIndexAtPosition(ann.getBegin());
      // fill previous section until here
      if(begin < cursor) throw new IllegalArgumentException("document is not properly annotated");
      for(int t = cursor; t < begin; t++) {
        array[t] = sectionIndex;
        cursor++;
      }
      if(enableMergeSections) {
        currentSection = ann.getSectionLabelOrHeading(); // merge same predictions
      } else {
        currentSection = Integer.toString(ann.getBegin());  // no merge
      }
      if(!currentSection.equals(lastSection)) {
        sectionIndex++; // merge sections with same name in gold segmentation
      }
      lastSection = currentSection;
    }
    // fill last section
    for(int t = cursor; t < array.length; t++) {
      array[t] = sectionIndex;
    }
    return array;
  }
  
}
