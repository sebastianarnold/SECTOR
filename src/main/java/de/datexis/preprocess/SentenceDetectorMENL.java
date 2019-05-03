package de.datexis.preprocess;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import de.datexis.common.WordHelpers;
import opennlp.tools.ml.model.MaxentModel;
import opennlp.tools.sentdetect.DefaultEndOfSentenceScanner;
import opennlp.tools.sentdetect.EndOfSentenceScanner;
import opennlp.tools.sentdetect.SDContextGenerator;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.util.Span;
import opennlp.tools.util.StringUtil;

/**
 * A sentence detector for splitting up raw text into sentences.
 * <p>
 * A maximum entropy model is used to evaluate end-of-sentence characters in a
 * string to determine if they signify the end of a sentence.
 */
public class SentenceDetectorMENL extends SentenceDetectorME {

  private MaxentModel model;
  private SDContextGenerator cgen;
  private EndOfSentenceScanner scanner;
  private List<Double> sentProbs = new ArrayList<>();
  
  public static final char[] newlineEosCharacters = new char[] { '.', '!', '?', '\n' };

  public SentenceDetectorMENL(SentenceModel model) {
    super(model);
    initializeFieldsFromReflection();
    this.scanner = new DefaultEndOfSentenceScanner(newlineEosCharacters);
  }
  
  private void initializeFieldsFromReflection() {
    try {
      // we need to get some fields from SentenceDetectorME through reflection
      Field modelField = SentenceDetectorME.class.getDeclaredField("model");
      modelField.setAccessible(true);
      this.model = (MaxentModel) modelField.get(this);
      Field cgenField = SentenceDetectorME.class.getDeclaredField("cgen");
      cgenField.setAccessible(true);
      this.cgen = (SDContextGenerator) cgenField.get(this);
      Field scannerField = SentenceDetectorME.class.getDeclaredField("scanner");
      scannerField.setAccessible(true);
      this.scanner = (EndOfSentenceScanner) scannerField.get(this);
      Field sentProbsField = SentenceDetectorME.class.getDeclaredField("sentProbs");
      sentProbsField.setAccessible(true);
      this.sentProbs = (List<Double>) sentProbsField.get(this);
    } catch(Exception ex) {
      Logger.getLogger(SentenceDetectorMENL.class.getName()).log(Level.SEVERE, null, ex);
    }
  }

  /**
   * Detect the position of the first words of sentences in a String.
   *
   * @param s  The string to be processed.
   * @return   A integer array containing the positions of the end index of
   *          every sentence
   *
   */
  @Override
  public Span[] sentPosDetect(String s) {
    sentProbs.clear();
    StringBuffer sb = new StringBuffer(s);
    List<Integer> enders = scanner.getPositions(s);
    List<Integer> positions = new ArrayList<>(enders.size());

    for (int i = 0, end = enders.size(), index = 0; i < end; i++) {
      int cint = enders.get(i);
      // skip over the leading parts of non-token final delimiters
      int fws = getFirstWS(s,cint + 1);
      if (i + 1 < end && enders.get(i + 1) < fws) {
        continue;
      }
      if (positions.size() > 0 && cint < positions.get(positions.size() - 1)) continue;

      double[] probs = model.eval(cgen.getContext(sb, cint));
      String bestOutcome = model.getBestOutcome(probs);
      
      // check if this or next char is a newline
      int nint = getFirstNonWS(s, cint + 1);
      if(nint < s.length() && s.charAt(nint) == '\n') bestOutcome = NO_SPLIT;
      if(s.charAt(cint) == '\n') bestOutcome = SPLIT;

      if (bestOutcome.equals(SPLIT) && isAcceptableBreak(s, index, cint)) {
        if (index != cint) {
          if (useTokenEnd && s.charAt(cint) != '\n') {
            positions.add(getFirstNonWS(s, getFirstWS(s,cint + 1)));
          } else {
            positions.add(getFirstNonWS(s, cint + 1));
          }
          sentProbs.add(probs[model.getIndex(bestOutcome)]);
        }

        index = cint + 1;
      }
    }

    int[] starts = new int[positions.size()];
    for (int i = 0; i < starts.length; i++) {
      starts[i] = positions.get(i);
    }

    // string does not contain sentence end positions
    if (starts.length == 0) {

      // remove leading and trailing whitespace
      int start = 0;
      int end = s.length();

      while (start < s.length() && StringUtil.isWhitespace(s.charAt(start)))
        start++;

      while (end > 0 && StringUtil.isWhitespace(s.charAt(end - 1)))
        end--;

      if (end - start > 0) {
        sentProbs.add(1d);
        return new Span[] {new Span(start, end)};
      }
      else
        return new Span[0];
    }

    // Convert the sentence end indexes to spans

    boolean leftover = starts[starts.length - 1] != s.length();
    Span[] spans = new Span[leftover ? starts.length + 1 : starts.length];

    for (int si = 0; si < starts.length; si++) {
      int start;

      if (si == 0) {
        start = 0;
      }
      else {
        start = starts[si - 1];
      }

      // A span might contain only white spaces, in this case the length of
      // the span will be zero after trimming and should be ignored.
      Span span = trimSpan(new Span(start, starts[si]), s);
      if (span.length() > 0) {
        spans[si] = span;
      }
//      else {
//        sentProbs.remove(si);
//      }
    }

    if (leftover) {
      Span span = trimSpan(new Span(starts[starts.length - 1], s.length()), s);
      if (span.length() > 0) {
        spans[spans.length - 1] = span;
        sentProbs.add(1d);
      }
    }
    /*
     * set the prob for each span
     */
    for (int i = 0; i < spans.length; i++) {
      if(spans[i] != null) {
        double prob = sentProbs.get(i);
        spans[i] = new Span(spans[i], prob);
      }
    }

    return spans;
  }

  /** trim span, but keep Newlines */
  public Span trimSpan(Span span, CharSequence text) {

    int newStartOffset = span.getStart();

    for (int i = span.getStart(); i < span.getEnd() && StringUtil.isWhitespace(text.charAt(i)) /*&& text.charAt(i) != '\n'*/; i++) {
      newStartOffset++;
    }

    int newEndOffset = span.getEnd();
    for (int i = span.getEnd(); i > span.getStart() && StringUtil.isWhitespace(text.charAt(i - 1)) && text.charAt(i - 1) != '\n'; i--) {
      newEndOffset--;
    }

    if (newStartOffset == span.getStart() && newEndOffset == span.getEnd()) {
      return span;
    } else if (newStartOffset > newEndOffset) {
      return new Span(span.getStart(), span.getStart(), span.getType());
    } else {
      return new Span(newStartOffset, newEndOffset, span.getType());
    }
  }
  
  /**
   * Allows subclasses to check an overzealous (read: poorly
   * trained) model from flagging obvious non-breaks as breaks based
   * on some boolean determination of a break's acceptability.
   *
   * <p>The implementation here always returns true, which means
   * that the MaxentModel's outcome is taken as is.</p>
   *
   * @param s the string in which the break occurred.
   * @param fromIndex the start of the segment currently being evaluated
   * @param candidateIndex the index of the candidate sentence ending
   * @return true if the break is acceptable
   */
  @Override
  protected boolean isAcceptableBreak(String s, int fromIndex, int candidateIndex) {
    if(s.length() < candidateIndex - 1) return true; // last position
    String test = s.substring(fromIndex, candidateIndex + 1);
    if(WordHelpers.abbreviationsEN.stream().anyMatch(abrv -> test.endsWith(abrv)) ||
      WordHelpers.abbreviationsDE.stream().anyMatch(abrv -> test.endsWith(abrv)))
      return false;
    return true;
  }

  private int getFirstWS(String s, int pos) {
    while (pos < s.length() && !StringUtil.isWhitespace(s.charAt(pos)))
      pos++;
    return pos;
  }

  private int getFirstNonWS(String s, int pos) {
    while (pos < s.length() && StringUtil.isWhitespace(s.charAt(pos)))
      pos++;
    return pos;
  }
  
}
