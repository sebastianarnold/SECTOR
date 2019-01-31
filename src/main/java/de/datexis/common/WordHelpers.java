package de.datexis.common;

import com.google.common.collect.Lists;
import de.datexis.model.Span;
import de.datexis.model.Token;
import java.io.IOException;
import java.text.Normalizer;
import java.util.*;
import java.util.regex.Pattern;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Utility class with static helpers for Strings
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class WordHelpers {

  protected final static Logger log = LoggerFactory.getLogger(WordHelpers.class);

  public static HashSet<String> skipSpaceBefore = new HashSet<>(Arrays.asList(",", ".", ":", ";", "?", "!", ")", "]", "'m", "'s", "'re", "'ve", "'d", "'ll", "n't"));
  public static HashSet<String> skipSpaceAfter = new HashSet<>(Arrays.asList("(", "[", "", "\n"));
  private static final String[][] umlautReplacements = { {"Ä","Ae"}, {"Ü","Ue"}, {"Ö","Oe"}, {"ä","ae"}, {"ü","ue"}, {"ö","oe"}, {"ß","ss"}, {"–","-"} };
  private static final String[][] tokenizationReplacements = { {"``","\""}, {"''","\""} };
  public static final Pattern punctPattern = Pattern.compile("[^\\w\\s\\-_]+");
  public static final Pattern spacePattern = Pattern.compile("[\\s]+");
  public static final Pattern numericPattern = Pattern.compile("[\\d]+");
  public static final Pattern bracketsPattern = Pattern.compile("[\\(\\)\\[\\]\"]");

  // TODO: umlaute fehlen hier!
  // Lists taken from: http://www.statmt.org/europarl/ tools
  public static HashSet<String> abbreviationsEN = new HashSet<>(Arrays.asList("Adj.", "Adm.", "Adv.", "Asst.", "Bart.", "Bldg.", "Brig.", "Bros.", "Capt.", "Cmdr.", "Col.",
    "Comdr.", "Con.", "Corp.", "Cpl.", "DR.", "Dr.", "Drs.", "Ens.", "Gen.", "Gov.", "Hon.", "Hr.", "Hosp.", "Insp.", "Lt.", "MM.", "MR.", "MRS.", "MS.", "Maj.", "Messrs.",
    "Mlle.", "Mme.", "Mr.", "Mrs.", "Ms.", "Msgr.", "Op.", "Ord.", "Pfc.", "Ph.", "Prof.", "Pvt.", "Rep.", "Reps.", "Res.", "Rev.", "Rt.", "Sen.", "Sens.", "Sfc.", "Sgt.",
    "Sr.", "St.", "Supt.", "Surg", "v.", "vs.", "i.e.", "rev.", "e.g.", "No.", "Nr.", "pp."));
  public static HashSet<String> abbreviationsDE = new HashSet<>(Arrays.asList("I.", "II.", "III.", "IV.", "V.", "VI.", "VII.", "VIII.", "IX.", "X.", "XI.", "XII.", "XIII.",
    "XIV.", "XV.", "XVI.", "XVII.", "XVIII.", "XIX.", "XX.", "i.", "ii.", "iii.", "iv.", "v.", "vi.", "vii.", "viii.", "ix.", "x.", "xi.", "xii.", "xiii.", "xiv.", "xv.",
    "xvi.", "xvii.", "xviii.", "xix.", "xx.", "Adj.", "Adm.", "Adv.", "Asst.", "Bart.", "Bldg.", "Brig.", "Bros.", "Capt.", "Cmdr.", "Col.", "Comdr.", "Con.", "Corp.",
    "Cpl.", "DR.", "Dr.", "Ens.", "Gen.", "Gov.", "Hon.", "Hosp.", "Insp.", "Lt.", "MM.", "MR.", "MRS.", "MS.", "Maj.", "Messrs.", "Mlle.", "Mme.", "Mr.", "Mrs.", "Ms.",
    "Msgr.", "Op.", "Ord.", "Pfc.", "Ph.", "Prof.", "Pvt.", "Rep.", "Reps.", "Res.", "Rev.", "Rt.", "Sen.", "Sens.", "Sfc.", "Sgt.", "Sr.", "St.", "Supt.", "Surg.",
    "Mio.", "Mrd.", "bzw.", "v.", "vs.", "usw.", "d.h.", "z.B.", "u.a.", "etc.", "Mrd.", "MwSt.", "ggf.", "d.J.", "D.h.", "m.E.", "vgl.", "I.F.", "z.T.", "sogen.", "ff.",
    "u.E.", "g.U.", "g.g.A.", "c.-à-d.", "Buchst.", "u.s.w.", "sog.", "u.ä.", "Std.", "evtl.", "Zt.", "Chr.", "u.U.", "o.ä.", "Ltd.", "b.A.", "z.Zt.", "spp.", "sen.",
    "SA.", "k.o.", "jun.", "i.H.v.", "dgl.", "dergl.", "Co.", "zzt.", "usf.", "s.p.a.", "Dkr.", "Corp.", "bzgl.", "BSE.", "No.", "Nos.", "Art.", "Nr.", "pp.", "ca.", "Ca"));

  public static enum Language { EN, DE };
  
  private final Set<String> stopWords;
  
  public WordHelpers(Language lang) {
    stopWords = new TreeSet<>(readStopWords(lang));
  }
  
  public static Language getLanguage(String language) {
    try {
      return Language.valueOf(language.trim().toUpperCase());
    } catch(IllegalArgumentException e) {
      return Language.EN;
    }
  }
  
  private List<String> readStopWords(Language lang) {
    Resource stop = Resource.fromJAR("stopwords/stopwords_" + lang.toString().toLowerCase() + ".csv");
    List<String> stopWords = new ArrayList<>();
    try {
      stopWords = IOUtils.readLines(stop.getInputStream(), "UTF-8");
    } catch (IOException ex) {
      log.error("Could not read stop words " + ex.toString());
    }
    return stopWords;
  }
  
  public List<String> getStopWords() {
    return Lists.newArrayList(stopWords);
  }
  
  public boolean isStopWord(String word, TokenPreProcess pre) {
    return isStopWord(pre.preProcess(word));
  }
  
  public boolean isStopWord(String word) {
    return stopWords.contains(word.toLowerCase());
  }
  
  /**
   * Builds a String from given words, with rule-based spacing according to characters.
   * @param tokens
   * @return
   */
  public static String wordsToText(Iterable<Token> tokens) {
    StringBuilder res = new StringBuilder();
    String last = "";
		for(Token t : tokens) {
      if(!skipSpaceAfter.contains(last) && !skipSpaceBefore.contains(t.getText())) res.append(" ");
 			res.append(t.getText());
      last = t.getText();
		}
		return res.toString().trim();
  }
  
  /**
   * Builds a String from given Tokens with their original spacing. If offsets are not assigned correctly, will return a space-seperated String.
   * @param tokens list of Tokens
   * @param beginOffset the start offset for this String. If 0, words will be padded until their original position.
   * @return the original Text, if possible
   */
  public static String tokensToText(Iterable<Token> tokens, int beginOffset) {
    StringBuilder res = new StringBuilder();
    int cursor = beginOffset;
    for(Token t : tokens) {
      if(t.isEmpty()) continue;
      if(cursor > t.getBegin()) {
        // reset in case of wrong offsets
        res.append(" ");
        cursor = t.getBegin();
      }
      while(cursor < t.getBegin()) {
        // append whitespace until begin is reached
        res.append(" ");
        cursor++;
      }
      // append text until end of token is reached.
      // This is important, because while Tokenization, "etc." could be converted into [etc]. [.] 
      final String word = t.getText();
      if(t.getLength() == word.length()) res.append(word);
      else if(t.getLength() < word.length()) res.append(word.substring(0, t.getLength())); // truncate word
      else res.append(word).append(String.join("", Collections.nCopies(t.getLength() - word.length(), " "))); // add spaces
      cursor = t.getEnd();
    }
		return res.toString();
  }
  
  /**
   * Cosine similarity between two vectors.
   * @return 1 for high similarity, 0 for orthogonal vectors, -1 for vectors pointing in the opposite direction
   * If one of both vectors is a Null-Vector or null, 0 is returned
   */
  public static double cosineSim(INDArray arr1, INDArray arr2) {
    if(arr1 == null || arr2 == null ) return 0;
    else if(arr1.sumNumber().doubleValue() == 0 || arr2.sumNumber().doubleValue() == 0 ) return 0;
    else return Transforms.cosineSim(arr1, arr2);
  }
  
  public static String vecToString(INDArray vec) {
    StringBuilder sb = new StringBuilder();
    for(int j = 0; j < vec.length(); j++) {
      sb.append(vec.getDouble(j));
      if(j < vec.length() - 1) {
        sb.append(" ");
      }
    }
    return sb.toString();
  }
  
  // TODO: this takes a lot of time?
  public static INDArray stringToVec(String str) {
    String[] split = str.split(" ");
    float[] vector = new float[split.length];
    for(int i = 0; i < split.length; i++) {
      vector[i] = Float.parseFloat(split[i]);
    }
    return Nd4j.create(vector);
  }
  
  public static String replaceAccents(String str) {
    //str = StringUtils.stripAccents(str);
    str = Normalizer.normalize(str, Normalizer.Form.NFD);
    //str = str.replaceAll("[\\p{InCombiningDiacriticalMarks}]", "");
    return str;
  }
  
  public static String replaceUmlauts(String str) {
    for(String[] rep : umlautReplacements) {
        str = str.replaceAll(rep[0], rep[1]);
    }
    return str;
  }
  
  public static String replacePunctuation(String str, String rep) {
    return punctPattern.matcher(str).replaceAll(rep);
  }
  
  public static String replaceNumbers(String str, String rep) {
    return numericPattern.matcher(str).replaceAll(rep);
  }
  
  public static String replaceSpaces(String str, String rep) {
    return spacePattern.matcher(str).replaceAll(rep);
  }
  
  public static String[] splitSpaces(String str) {
    return str.split(spacePattern.pattern());
  }
    
  public static int getSpanOverlapLength(Span a, Span b) {
    int begin = Math.max(a.getBegin(), b.getBegin());
    int end = Math.min(a.getEnd(), b.getEnd());
    if(begin < end) return end - begin;
    else return 0;
  }
  
  
}
