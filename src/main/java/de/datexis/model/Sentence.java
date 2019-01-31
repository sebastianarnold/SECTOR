package de.datexis.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import java.util.ArrayList;
import java.util.List;

import de.datexis.common.WordHelpers;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * A Sentence is a sequence of Tokens. It may contain the original untokenized plain text if needed.
 * @author sarnold, fgrimme
 */
public class Sentence extends Span {

	/**
	 * List of all tokens referenced by this Sentence instance
	 */
	private List<Token> tokens;
  
	/**
	 * Create an empty Sentence
	 */
	public Sentence() {
		tokens = new ArrayList<>();
	}
	
	/**
	 * Constructor using fields
	 * 
	 * @param tokens
	 *            - Tokens to be referenced by this instance
	 */
	public Sentence(List<Token> tokens) {
		super();
		setTokens(tokens);
	}
  
	/**
	 * Iterate over all Tokens in this Sentence
	 * 
	 * @return
	 */
  @JsonIgnore
	public List<Token> getTokens() {
		return tokens;
	}
  
  public Stream<Token> streamTokens() {
    return tokens.stream();
  }
  
  @JsonIgnore
  public boolean isEmpty() {
    return tokens.isEmpty();
  }
  
  /**
   * Returns the i-th Token of this Sentence
   * @param offset starting with S0
   * @return 
   */
  public Token getToken(int offset) {
    return tokens.get(offset);
  }
  
  public int countTokens() {
    return tokens.size();
  }

	/**
	 * Set Tokens
	 * @param t
	 *            - Tokens to be referenced by this Sentence instance
	 */
	public void setTokens(List<Token> t) {
		tokens = t;
    if(!t.isEmpty()) {
      begin = tokens.get(0).getBegin();
      end = tokens.get(tokens.size() - 1).getEnd();
    } else {
      begin = 0;
      end = 0;
    }
	}

	public void addToken(Token t) {
    if(tokens.isEmpty()) begin = t.getBegin();
    end = t.getEnd();
		tokens.add(t);
	}
    
	/**
	 * Functional access to all Tokens in this Sentence
	 */
	public static com.google.common.base.Function<Sentence, Iterable<Token>> getTokens = (Sentence s) -> s.tokens;

  @Override
	public String toString() {
    return toTokenizedString();
  }
  
	/**
	 * Returns all Tokens of the Sentence seperated with spaces Please note: the
	 * original Document might have different whitespace segmentation
	 * 
	 * @return
	 */
	public String toTokenizedString() {
		return streamTokens().map(t -> t.getText()).collect(Collectors.joining(" "));
	}
  
  /**
   * Returns the original Text of the sentence with correct segmentation.
   * @return 
   */
  @Override
  public String getText() {
    return WordHelpers.tokensToText(getTokens(), getBegin());
  }

  /**
   * @return a deep copy of this Sentence
   */
  public Sentence clone() {
    return clone(Function.identity());
  }
  
  public Sentence clone(Function<String,String> transform) {
    Sentence result = new Sentence();
    for(Token t : getTokens()) {
      result.addToken(t.clone(transform));
    }
    result.setBegin(getBegin());
    result.setEnd(getEnd());
    return result;
  }

  void addOffset(int offset) {
    int length = getLength();
    setBegin(getBegin() + offset);
    setLength(length);
    for(Token t : tokens) {
      length = t.getLength();
      t.setBegin(t.getBegin() + offset);
      t.setLength(length);
    }
  }

  @Override
  public boolean equals(Object o) {
    if(this == o) {
      return true;
    }
    if(!(o instanceof Sentence)) {
      return false;
    }
    Sentence sentence = (Sentence) o;
    
    return super.equals(sentence) &&
           Objects.equals(getText(), sentence.getText()) &&
           Objects.equals(getTokens(), sentence.getTokens());
  }

  @Override
  public int hashCode() {
    return Objects.hash(super.hashCode(), getTokens(), getText());
  }
}
