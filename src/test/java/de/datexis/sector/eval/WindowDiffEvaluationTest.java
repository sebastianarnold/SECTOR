package de.datexis.sector.eval;

import de.datexis.sector.eval.SegmentationEvaluation;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;

import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.sector.model.SectionAnnotation;

import org.jetbrains.annotations.NotNull;
import org.junit.Before;

import static org.hamcrest.Matchers.*;
import static org.junit.Assert.*;

import org.junit.Test;

import java.util.*;


public class WindowDiffEvaluationTest {

  private static final String DEFAULT_TEST_TYPE = "DEFAULT_TEST_TYPE";
  private static final String DEFAULT_TEST_SECTION = "DEFAULT_TEST_SECTION";

  private static final Double ALL_BOUNDARIES_CORRECT_SCORE = 0d;

  private static final double ACCEPTED_ERROR_MARGIN = 0.03d;

  private static final int HEARST_DATASET_PAPER_VARIANT = 1;
  private static final int HEARST_DATASET_JUDGE_1_VARIANT = 2;
  private static final int HEARST_DATASET_JUDGE_2_VARIANT = 3;
  private static final int HEARST_DATASET_JUDGE_1_TEXOO_STYLE = 4;

  private static final int TEXOO_ANNOTATION_BOUNDARY_STYLE = 1;
  private static final int SEG_EVAL_BOUNDARY_STYLE = 2;

  public SegmentationEvaluation windowDiffEvaluation;

  @Before
  public void setUp() throws Exception {
    windowDiffEvaluation = new SegmentationEvaluation("TestRun");
  }

  @Test
  public void calculateWindowDiffShouldNotThrowException()
    throws Exception {
    Document documentWithTwoSentences = createDocumentWithNSentences(2);
    SectionAnnotation sectionAnnotation = createSectionAnnotation(0, 0, Annotation.Source.GOLD);
    documentWithTwoSentences.addAnnotation(sectionAnnotation);
    
    windowDiffEvaluation.calculateScores(Lists.newArrayList(documentWithTwoSentences));
  }

  @Test
  public void returnZeroWhenSegmentationAnnotationsAreIdentical() {
    Document documentWithTwoSentences = createDocumentWithNSentences(2);

    documentWithTwoSentences.addAnnotations(createDefaultSegmentationAnnotationList(Annotation.Source.GOLD));
    documentWithTwoSentences.addAnnotations(createDefaultSegmentationAnnotationList(Annotation.Source.PRED));

    windowDiffEvaluation.calculateScores(Lists.newArrayList(documentWithTwoSentences));
    
    assertEquals(ALL_BOUNDARIES_CORRECT_SCORE, windowDiffEvaluation.getWD(), 1e-5);
  }

  @Test
  public void returnNotZeroWhenNoAnnotationsWerePredicted() {
    Document documentWithThreeSentences = createDocumentWithNSentences(3);

    documentWithThreeSentences.addAnnotations(createDefaultSegmentationAnnotationList(Annotation.Source.GOLD));

    windowDiffEvaluation.calculateScores(Lists.newArrayList(documentWithThreeSentences));

    assertThat(windowDiffEvaluation.getWD(), is(greaterThan(ALL_BOUNDARIES_CORRECT_SCORE)));
  }

  @Test
  public void returnNotZeroWhenAnnotationsAreNotIdentical() {
    Document documentWithTwoSentences = createDocumentWithNSentences(5);

    documentWithTwoSentences.addAnnotations(createDefaultSegmentationAnnotationList(Annotation.Source.GOLD));
    documentWithTwoSentences.addAnnotations(createSegmentationAnnotationList(Sets.newHashSet(0, 1, 2, 3), Annotation.Source.PRED));

    windowDiffEvaluation.calculateScores(Lists.newArrayList(documentWithTwoSentences));

    assertThat(windowDiffEvaluation.getWD(), is(greaterThan(ALL_BOUNDARIES_CORRECT_SCORE)));
  }

  @Test
  public void noGoldSegmentsOn13Sentences() {
    Document document = createDocumentWithNSentences(13);
    document.addAnnotation(createSectionAnnotation(0, 12, Annotation.Source.GOLD));
    document.addAnnotation(createSectionAnnotation(0, 3, Annotation.Source.PRED));
    document.addAnnotation(createSectionAnnotation(4, 7, Annotation.Source.PRED));
    document.addAnnotation(createSectionAnnotation(8, 12, Annotation.Source.PRED));

    windowDiffEvaluation.calculateScores(Lists.newArrayList(document));

    assertThat(windowDiffEvaluation.getWD(), is(closeTo(1d, ACCEPTED_ERROR_MARGIN)));
  }

  @Test
  public void threeGoldSegmentsOn13Sentences() {
    Document document = createDocumentWithNSentences(13);

    SectionAnnotation predictedAnnotation = createSectionAnnotation(0, 12, Annotation.Source.PRED);
    document.addAnnotation(createSectionAnnotation(0, 3, Annotation.Source.GOLD));
    document.addAnnotation(createSectionAnnotation(4, 7, Annotation.Source.GOLD));
    document.addAnnotation(createSectionAnnotation(8, 12, Annotation.Source.GOLD));
    document.addAnnotation(predictedAnnotation);

    windowDiffEvaluation.calculateScores(Lists.newArrayList(document));

    assertThat(windowDiffEvaluation.getWD(), is(closeTo(0.363636d, ACCEPTED_ERROR_MARGIN)));
  }

  @Test
  public void threeGoldSegmentsOn13SentencesOneAdditionalSegmentPredicted() {
    Document document = createDocumentWithNSentences(13);

    document.addAnnotation(createSectionAnnotation(0, 4, Annotation.Source.PRED));
    document.addAnnotation(createSectionAnnotation(5, 5, Annotation.Source.PRED));
    document.addAnnotation(createSectionAnnotation(6, 7, Annotation.Source.PRED));
    document.addAnnotation(createSectionAnnotation(8, 12, Annotation.Source.PRED));
    document.addAnnotation(createSectionAnnotation(0, 4, Annotation.Source.GOLD));
    document.addAnnotation(createSectionAnnotation(5, 7, Annotation.Source.GOLD));
    document.addAnnotation(createSectionAnnotation(8, 12, Annotation.Source.GOLD));

    windowDiffEvaluation.calculateScores(Lists.newArrayList(document));

    assertThat(windowDiffEvaluation.getWD(), is(closeTo(0.181818d, ACCEPTED_ERROR_MARGIN)));
  }

  @Test
  public void threeGoldSegmentsOn13SentencesOneAdditionalSegmentAllMisalignedPredicted() {
    Document document = createDocumentWithNSentences(13);

    document.addAnnotation(createSectionAnnotation(0, 5, Annotation.Source.PRED));
    document.addAnnotation(createSectionAnnotation(6, 6, Annotation.Source.PRED));
    document.addAnnotation(createSectionAnnotation(7, 8, Annotation.Source.PRED));
    document.addAnnotation(createSectionAnnotation(9, 12, Annotation.Source.PRED));
    document.addAnnotation(createSectionAnnotation(0, 4, Annotation.Source.GOLD));
    document.addAnnotation(createSectionAnnotation(5, 7, Annotation.Source.GOLD));
    document.addAnnotation(createSectionAnnotation(8, 12, Annotation.Source.GOLD));

    windowDiffEvaluation.calculateScores(Lists.newArrayList(document));

    assertThat(windowDiffEvaluation.getWD(), is(closeTo(0.272727d, ACCEPTED_ERROR_MARGIN)));
  }

  @Test
  public void testOnHearst1997DataSetOnJudgeOne() {
    testHearst1997DataSet(0.31578947d, HEARST_DATASET_JUDGE_1_VARIANT, SEG_EVAL_BOUNDARY_STYLE);
  }

  @Test
  public void testOnHearst1997DataSetOnJudgeOneStylesShouldMakeNoDifference() {
    testHearst1997DataSet(0.31578947d, HEARST_DATASET_JUDGE_1_VARIANT, SEG_EVAL_BOUNDARY_STYLE);
    testHearst1997DataSet(0.31578947d, HEARST_DATASET_JUDGE_1_TEXOO_STYLE, TEXOO_ANNOTATION_BOUNDARY_STYLE);
  }

  @Test
  public void testOnHearst1997DataSetOnJudgeTwo() {
    testHearst1997DataSet(0.42105263d, HEARST_DATASET_JUDGE_2_VARIANT, TEXOO_ANNOTATION_BOUNDARY_STYLE);
  }

  private void testHearst1997DataSet(double result, int variant, int annotationBoundaryStyle) {
    Document document = createDocumentWithNSentences(21);
    document.addAnnotations(createHearstDataSetGoldAnnotations(variant));
    document.addAnnotations(createHearstDataSetPredictedAnnotations(annotationBoundaryStyle));

    windowDiffEvaluation.calculateScores(Lists.newArrayList(document));

    assertThat(windowDiffEvaluation.getWD(), is(closeTo(result, ACCEPTED_ERROR_MARGIN)));
  }

  @Test
  public void scoreShouldNotDifferWithAndWithoutWhiteSpaceBetweenSentences() {
    int numSentences = 70;
    
    Document documentWithWhiteSpace = setUpDocumentForWhiteSpaceTest(numSentences, true);
    documentWithWhiteSpace = addGoldAnnotationsForWitheSpaceTest(documentWithWhiteSpace);
    documentWithWhiteSpace = addPredictedAnnotationsForWitheSpaceTest(documentWithWhiteSpace);
    
    Document documentWithoutWhiteSpace = setUpDocumentForWhiteSpaceTest(numSentences, false);
    documentWithoutWhiteSpace = addGoldAnnotationsForWitheSpaceTest(documentWithoutWhiteSpace);
    documentWithoutWhiteSpace = addPredictedAnnotationsForWitheSpaceTest(documentWithoutWhiteSpace);
    
    windowDiffEvaluation.calculateScores(Lists.newArrayList(documentWithoutWhiteSpace));
    SegmentationEvaluation windowDiffEvaluationWithWhitespace = new SegmentationEvaluation("white space");
    windowDiffEvaluationWithWhitespace.calculateScores(Lists.newArrayList(documentWithWhiteSpace));

    assertThat(windowDiffEvaluation.getWD(), is(closeTo(windowDiffEvaluationWithWhitespace.getWD(), ACCEPTED_ERROR_MARGIN)));
  }

  private Document addGoldAnnotationsForWitheSpaceTest(Document document) {
    SectionAnnotation section1 = createSectionSpanningOverSentences(document, Annotation.Source.GOLD, 0, 19);
    SectionAnnotation section2 = createSectionSpanningOverSentences(document, Annotation.Source.GOLD, 20, 49);
    SectionAnnotation section3 = createSectionSpanningOverSentences(document, Annotation.Source.GOLD, 50, 69);

    document.addAnnotation(section1);
    document.addAnnotation(section2);
    document.addAnnotation(section3);

    return document;
  }

  private Document addPredictedAnnotationsForWitheSpaceTest(Document document) {
    SectionAnnotation section1 = createSectionSpanningOverSentences(document, Annotation.Source.PRED, 0, 25);
    SectionAnnotation section2 = createSectionSpanningOverSentences(document, Annotation.Source.PRED, 26, 40);
    SectionAnnotation section3 = createSectionSpanningOverSentences(document, Annotation.Source.PRED, 41, 69);

    document.addAnnotation(section1);
    document.addAnnotation(section2);
    document.addAnnotation(section3);

    return document;
  }

  private SectionAnnotation createSectionSpanningOverSentences(Document document,
                                                               Annotation.Source source,
                                                               int beginSentenceIndex,
                                                               int endSentenceIndex) {
    int sectionBegin = document.getSentence(beginSentenceIndex).getBegin();
    int sectionEnd = document.getSentence(endSentenceIndex).getEnd();

    SectionAnnotation section1 = new SectionAnnotation(source);
    section1.setBegin(sectionBegin);
    section1.setEnd(sectionEnd);
    section1.setSectionHeading(Integer.toString(sectionBegin) + "-" + Integer.toString(sectionEnd));

    return section1;
  }

  private Document setUpDocumentForWhiteSpaceTest(int numSentences, boolean addWhiteSpace) {
    Document document = new Document();
    int positionMultiplier;

    if(addWhiteSpace) {
      positionMultiplier = 2;
    } else {
      positionMultiplier = 1;
    }

    for(int i = 0; i < numSentences; i++) {
      Sentence sentence = new Sentence();
      if(i == 0) {
        sentence.setBegin(i);
      } else {
        sentence.setBegin(i * positionMultiplier);
      }
      sentence.setEnd((i * positionMultiplier) + 1);
      document.addSentence(sentence, false);
    }

    return document;
  }

  private void addWhiteSpace(int correction, Span spanToCorrect) {
    spanToCorrect.setBegin(spanToCorrect.getBegin() + correction);
    spanToCorrect.setEnd(spanToCorrect.getEnd() + correction);
  }


  private List<SectionAnnotation> createHearstDataSetGoldAnnotations(int variant) {
    List<SectionAnnotation> goldAnnotations = Collections.EMPTY_LIST;

    switch(variant) {
      case HEARST_DATASET_PAPER_VARIANT:
      default:
        goldAnnotations = createHearstDataSetGoldAnnotationsAccordingToPaper();
        break;
      case HEARST_DATASET_JUDGE_1_VARIANT:
        goldAnnotations = createHearstDataSetGoldAnnotationsAccordingToJudgeOne();
        break;
      case HEARST_DATASET_JUDGE_2_VARIANT:
        goldAnnotations = createHearstDataSetGoldAnnotationsAccordingToJudgeTwo();
        break;
      case HEARST_DATASET_JUDGE_1_TEXOO_STYLE:
        goldAnnotations = createHearstDataSetGoldAnnotationsAccordingToJudgeOneInTeXooStyle();
    }

    return goldAnnotations;
  }

  /**
   * Boundaries according to judge Two: 2	10	12	16	18	21
   */
  private List<SectionAnnotation> createHearstDataSetGoldAnnotationsAccordingToJudgeTwo() {
    List<SectionAnnotation> goldAnnotations = new ArrayList<>(6);

    goldAnnotations.add(createSectionAnnotation(0, 2, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(3, 10, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(11, 12, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(13, 16, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(17, 18, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(19, 21, Annotation.Source.GOLD));

    return goldAnnotations;
  }


  /**
   * End positions of gold section labels: 2	3	5	8	9	12	13	16	18	21
   * Chooses as described in (Hearst,1997) from original data set (Hearst Stargazer).
   * Dataset is available at:
   * https://github.com/cfournie/segmentation.evaluation/tree/master/segeval
   *
   * @return List of gold SectionAnnotations corresponding to Hearst 1997
   */
  private List<SectionAnnotation> createHearstDataSetGoldAnnotationsAccordingToPaper() {
    List<SectionAnnotation> goldAnnotations = new ArrayList<>(10);

    goldAnnotations.add(createSectionAnnotation(0, 2, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(3, 3, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(4, 5, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(6, 8, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(9, 9, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(10, 12, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(13, 16, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(17, 18, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(19, 21, Annotation.Source.GOLD));

    return goldAnnotations;
  }


  /**
   * Boundaries according to judge one: 2	5	8	9	12	18	21
   *
   * @return gold labeled {@link SectionAnnotation}s according to judge one (Hearst, 1997)
   */
  private List<SectionAnnotation> createHearstDataSetGoldAnnotationsAccordingToJudgeOne() {
    List<SectionAnnotation> goldAnnotations = new ArrayList<>(7);

    goldAnnotations.add(createSectionAnnotation(0, 2, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(3, 5, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(6, 8, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(9, 9, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(10, 12, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(13, 18, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(19, 21, Annotation.Source.GOLD));


    return goldAnnotations;

  }

  /**
   * Boundaries according to judge one: 2	5	8	9	12	18	21
   *
   * @returngold labeled {@link SectionAnnotation}s according to judge one (Hearst, 1997) in TeXoo
   * annotation boundary style.
   */
  private List<SectionAnnotation> createHearstDataSetGoldAnnotationsAccordingToJudgeOneInTeXooStyle() {
    List<SectionAnnotation> goldAnnotations = new ArrayList<>(7);

    goldAnnotations.add(createSectionAnnotation(0, 2, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(2, 5, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(5, 8, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(8, 9, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(9, 12, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(12, 18, Annotation.Source.GOLD));
    goldAnnotations.add(createSectionAnnotation(18, 21, Annotation.Source.GOLD));


    return goldAnnotations;

  }

  /**
   * 2	8	12	14	18	21
   *
   * @return predicted labeled {@link SectionAnnotation}s according to (Hearst, 1997)
   */
  private List<SectionAnnotation> createHearstDataSetPredictedAnnotations(int style) {
    List<SectionAnnotation> predictedAnnotations = Collections.emptyList();

    switch(style) {
      case TEXOO_ANNOTATION_BOUNDARY_STYLE:
        predictedAnnotations = createHearstDataSetPredictedAnnotationsTeXooStyle();
        break;
      case SEG_EVAL_BOUNDARY_STYLE:
      default:
        predictedAnnotations = createHearstDataSetPredictedAnnotationsSegEvalStyle();
        break;
    }
    return predictedAnnotations;
  }

  private List<SectionAnnotation> createHearstDataSetPredictedAnnotationsSegEvalStyle() {
    List<SectionAnnotation> predictedAnnotations = new ArrayList<>(10);

    predictedAnnotations.add(createSectionAnnotation(0, 2, Annotation.Source.PRED));
    predictedAnnotations.add(createSectionAnnotation(3, 8, Annotation.Source.PRED));
    predictedAnnotations.add(createSectionAnnotation(9, 12, Annotation.Source.PRED));
    predictedAnnotations.add(createSectionAnnotation(13, 14, Annotation.Source.PRED));
    predictedAnnotations.add(createSectionAnnotation(15, 18, Annotation.Source.PRED));
    predictedAnnotations.add(createSectionAnnotation(19, 21, Annotation.Source.PRED));

    return predictedAnnotations;
  }

  private List<SectionAnnotation> createHearstDataSetPredictedAnnotationsTeXooStyle() {
    List<SectionAnnotation> predictedAnnotations = new ArrayList<>(10);

    predictedAnnotations.add(createSectionAnnotation(0, 2, Annotation.Source.PRED));
    predictedAnnotations.add(createSectionAnnotation(2, 8, Annotation.Source.PRED));
    predictedAnnotations.add(createSectionAnnotation(8, 12, Annotation.Source.PRED));
    predictedAnnotations.add(createSectionAnnotation(12, 14, Annotation.Source.PRED));
    predictedAnnotations.add(createSectionAnnotation(14, 18, Annotation.Source.PRED));
    predictedAnnotations.add(createSectionAnnotation(18, 21, Annotation.Source.PRED));

    return predictedAnnotations;
  }

  private Document createDocumentWithNSentences(int n) {
    Document document = new Document();
    for(int i = 0; i < n; i++) {
      document.addSentence(new Sentence());
    }
    return document;
  }

  @NotNull
  private List<SectionAnnotation> createSegmentationAnnotationList(Set<Integer> setOfBegins, Annotation.Source source) {
    Map<Integer, Integer> beginEndMap = Maps.asMap(setOfBegins, input -> input);
    return createSectionAnnotationList(beginEndMap, source);
  }

  @NotNull
  private List<SectionAnnotation> createDefaultSegmentationAnnotationList(Annotation.Source source) {
    HashSet<Integer> setOfBegins = Sets.newHashSet(0, 1);
    return createSegmentationAnnotationList(setOfBegins, source);
  }

  @NotNull
  private List<SectionAnnotation> createSectionAnnotationList(Map<Integer, Integer> beginEndMap, Annotation.Source source) {
    Iterator<Map.Entry<Integer, Integer>> entryIterator = beginEndMap.entrySet().iterator();
    List<SectionAnnotation> sectionAnnotationList = new ArrayList<>(0);

    if(beginEndMap.size() == 1) {
      return createSectionAnnotationListWithOneEntry(source, entryIterator);
    }

    SectionAnnotation[] sectionAnnotations = new SectionAnnotation[beginEndMap.size() - 1];

    for(int i = 0; entryIterator.hasNext(); i++) {
      Map.Entry<Integer, Integer> beginEndEntry = entryIterator.next();
      if(!entryIterator.hasNext()) {
        sectionAnnotationList = Lists.asList(
          createSectionAnnotation(beginEndEntry.getKey(), beginEndEntry.getValue(), source),
          sectionAnnotations);
        continue;
      }
      sectionAnnotations[i] = createSectionAnnotation(beginEndEntry.getKey(), beginEndEntry.getValue(), source);
    }

    return sectionAnnotationList;
  }

  @NotNull
  private List<SectionAnnotation> createSectionAnnotationListWithOneEntry(Annotation.Source source, Iterator<Map.Entry<Integer, Integer>> entryIterator) {
    List<SectionAnnotation> sectionAnnotations1;
    Map.Entry<Integer, Integer> onlyExistingEntry = entryIterator.next();
    SectionAnnotation sectionAnnotation = createSectionAnnotation(onlyExistingEntry.getKey(),
                                                                  onlyExistingEntry.getValue(),
                                                                  source);
    sectionAnnotations1 = Lists.newArrayList(sectionAnnotation);
    return sectionAnnotations1;
  }

  private SectionAnnotation createSectionAnnotation(int begin, int end, Annotation.Source source) {
    SectionAnnotation sectionAnnotation = new SectionAnnotation(source,
                                                                DEFAULT_TEST_TYPE,
                                                               Integer.toString(begin) + "-" + Integer.toString(end));
    sectionAnnotation.setBegin(begin);
    sectionAnnotation.setEnd(end);

    return sectionAnnotation;
  }

}
