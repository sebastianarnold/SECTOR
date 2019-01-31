package de.datexis.encoder.impl;

import de.datexis.encoder.StaticEncoder;
import de.datexis.model.Span;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.slf4j.LoggerFactory;

/**
 * Encodes several features of the word's surface form. Please note that we do not encode any
 * language specific features, such as closed word classes. These classes should be detected
 * by SkipGramEncoder depending on the language.
 * Some Penn Treebank Part-of-Speech Tags overviews and extensions can be found here:
 * https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
 * http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/Penn-Treebank-Tagset.pdf
 * https://www.eecis.udel.edu/~vijay/cis889/ie/pos-set.pdf
 * https://www.comp.leeds.ac.uk/ccalas/tagsets/upenn.html
 * http://erwinkomen.ruhosting.nl/eng/2014_Longdale-Labels.htm
 * @author sarnold
 */
public class SurfaceEncoder extends StaticEncoder {
  
  public SurfaceEncoder() {
    super("SUR");
    log = LoggerFactory.getLogger(SurfaceEncoder.class);
  }
  
  public SurfaceEncoder(String id) {
    super(id);
    log = LoggerFactory.getLogger(SurfaceEncoder.class);
  }
  
  @Override
  public String getName() {
    return "Surface Form Encoder";
  }

  @Override
  @JsonIgnore
  public long getEmbeddingVectorSize() {
    return encode("Test").length();
  }

  public void setVectorSize(int size) {
    if(size != getEmbeddingVectorSize()) {
      throw new IllegalArgumentException("Vector size of saved Encoder (" + getEmbeddingVectorSize() + ") differs from implementation (" + size + ")");
    }
  }
  
  @Override
  public INDArray encode(Span span) {
    return encode(span.getText());
  }
  
  @Override
  public INDArray encode(String span) {
    span = span.trim();
    ArrayList<Boolean> features = new ArrayList<>();
    // surface form features
    features.add(startsWithUppercase(span));
    features.add(startsWithLowercase(span));
    features.add(isAllUppercase(span));
    features.add(isAllLowercase(span));
    features.add(isMixedCase(span));
    features.add(isAllNumeric(span));
    features.add(includesNumeric(span));
    features.add(startsWithNumeric(span));
    features.add(endsWithNumeric(span));
    features.add(startsWithPunctuation(span));
    features.add(endsWithPunctuation(span));
    
    INDArray vector = Nd4j.zeros(features.size(), 1);
    int i = 0;
    for(Boolean f : features) {
      vector.put(i++, 0, f ? 1.0 : 0.0);
    }
    return vector;
  }
  
  // empty
  public boolean isEmpty(String token) {
    return token.isEmpty();
  }  
  // single char
  public boolean is1Char(String token) {
    return token.length() == 1;
  }  
  // two chars
  public boolean is2Chars(String token) {
    return token.length() == 2;
  }  
  // three chars
  public boolean is3Chars(String token) {
    return token.length() == 3;
  }
  // four chars
  public boolean is4Chars(String token) {
    return token.length() == 4;
  }
  // over four chars
  public boolean isOver4Chars(String token) {
    return token.length() > 4;
  }
  // over 12 chars (why?)
  public boolean isOver12Chars(String token) {
    return token.length() > 12;
  }
  // starts with uppercase
  public boolean startsWithUppercase(String token) {
    token = token.replaceAll("[^\\p{L}]", "");
    if(token.isEmpty()) return false;
    else token = token.substring(0, 1);
    return token.toUpperCase().equals(token);
  }
  // starts with lowercase
  public boolean startsWithLowercase(String token) {
    token = token.replaceAll("[^\\p{L}]", "");
    if(token.isEmpty()) return false;
    else token = token.substring(0, 1);
    return token.toLowerCase().equals(token);
  }
  // is all uppercase
  public boolean isAllUppercase(String token) {
    token = token.replaceAll("[^\\p{L}]", "");
    if(token.isEmpty()) return false;
    return token.toUpperCase().equals(token);
  }
  // is all lowercase
  public boolean isAllLowercase(String token) {
    token = token.replaceAll("[^\\p{L}]", "");
    return token.toLowerCase().equals(token);
  }
  // is mixed case
  public boolean isMixedCase(String token) {
    return !startsWithUppercase(token) && !isAllUppercase(token) && !isAllLowercase(token);
  }
  // is numeric
  // numeral and cardinal CD  0123456789
  public boolean isAllNumeric(String token) {
    return token.equals(token.replaceAll("[^\\p{N}\\p{P}]", ""));
  }
  // includes numeric
  public boolean includesNumeric(String token) {
    return !token.replaceAll("[^\\p{N}\\p{P}]", "").isEmpty();
  }
  // starts with numeric
  public boolean startsWithNumeric(String token) {
    if(token.isEmpty()) return false;
    else token = token.substring(0, 1);
    return token.equals(token.replaceAll("[^\\p{N}\\p{P}]", ""));
  }
  // ends with numeric
  public boolean endsWithNumeric(String token) {
    if(token.isEmpty()) return false;
    else token = token.substring(token.length()-1);
    return token.equals(token.replaceAll("[^\\p{N}\\p{P}]", ""));
  }
  // starts with punctuation
  public boolean startsWithPunctuation(String token) {
    if(token.isEmpty()) return false;
    else token = token.substring(0, 1);
    return token.equals(token.replaceAll("[^\\p{P}]", ""));
  }
  // ends with punctuation
  public boolean endsWithPunctuation(String token) {
    if(token.isEmpty()) return false;
    else token = token.substring(token.length()-1);
    return token.equals(token.replaceAll("[^\\p{P}]", ""));
  }
  // symbols and signs    SYM $ #
  static Collection<String> symbols = Arrays.asList("#","$","%","@","^","_","~","¢","£","¥","§","€");
  public boolean isSymbol(String token) {
    return symbols.contains(token);
  }
  // operators  + - & *
  static Collection<String> operators = Arrays.asList("&","*","+","=");
  public boolean isOperator(String token) {
    return operators.contains(token);
  }
  // opening quotes           ` ``
  // other quotes             "
  static Collection<String> oquotes = Arrays.asList("\"","`","``");
  public boolean isOpeningQuote(String token) {
    return symbols.contains(token);
  }
  // closing quotes           ' ''
  static Collection<String> cquotes = Arrays.asList("'","''");
  public boolean isClosingQuote(String token) {
    return symbols.contains(token);
  }
  // opening paranthesis      ( [ {
  // -LRB- -END-?
  static Collection<String> oparanthesis = Arrays.asList("(","<","[","{","-LRB-");
  public boolean isOpeningParanthesis(String token) {
    return oparanthesis.contains(token);
  }
  // closing paranthesis      ) ] }
  static Collection<String> cparanthesis = Arrays.asList(")",">","]","}","-RRB-");
  public boolean isClosingParanthesis(String token) {
    return cparanthesis.contains(token);
  }
  // slashes                  / \ |
  static Collection<String> slashes = Arrays.asList("/","\\","|");
  public boolean isSlash(String token) {
    return slashes.contains(token);
  }
  // comma                    ,
  static Collection<String> commas = Arrays.asList(",");
  public boolean isComma(String token) {
    return commas.contains(token);
  }
  // dashes                   -- - – 
  // list item marker     LS
  static Collection<String> dashes = Arrays.asList("-","–","--","---");
  public boolean isDash(String token) {
    return dashes.contains(token);
  }
  // sentence terminator      . ! ? 
  static Collection<String> sterminator = Arrays.asList(".","!","?");
  public boolean isSentenceTerminator(String token) {
    return sterminator.contains(token);
  }
  // colons and ellipses      : ; ... 
  static Collection<String> colons = Arrays.asList(":",";","...");
  public boolean isColon(String token) {
    return colons.contains(token);
  }

  // don't encode any closed word classes (language dependent!):
  // conjunction          IN
  // determiner           DT
  // preposition          IN
  // possessive ending    POS
  // wh-pronoun           W*
  
}
