package de.datexis.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import de.datexis.model.tag.Tag;

import java.util.Objects;
import java.util.function.Function;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A Token is an atomic entry in a Sentence. This implementation uses Stanford Words.
 * @author sarnold, fgrimme
 */
public class Token extends Span {

  protected static final Logger log = LoggerFactory.getLogger(Token.class);
    
  /**
   * The text of this Token
   */
  private String text;
    
  /**
   * Default constructor.
   * @deprecated only used for JSON deserialization.
   */
  @Deprecated
  protected Token() {}
  
  /**
   * Create a Token with given text and position
   * @param text 
   * @param begin 
   * @param end 
   */
  public Token(String text, int begin, int end) {
    // TODO: maybe include a field for token Value like -LRB- -END-?
    this.text = text;
    this.begin = begin;
    this.end = end;
  }
  
  /**
   * Create a Token with given text
   * @param text 
   */
  public Token(String text) {
    this(text, 0, text.length());
  }
  
  /**
   * Returns the text of this Token
   * @return 
   */
  @Override
  public String getText() {
    return text;
  }

  public void setText(String text) {
    this.text = text;
  }
  
  /**
   * Returns the cursor position of the beginning of the Token
   * @return 
   */
  @Override
	public int getBegin() {
		return begin;
	}
	
  /**
   * Returns the cursor position of the end of the Token (exclusive)
   * @return 
   */
  @Override
	public int getEnd() {
		return end;
	}

	@Override
	public String toString() {
    return getText();
	}
  
  @JsonIgnore
  public boolean isEmpty() {
    return text.isEmpty();
  }
  
  /**
   * Convenience method to put a Tag on a Token.
   * @param <T>
   * @param source
   * @param tag
   * @return The Token itself for function chaining.
   */
  @Override
  public <T extends Tag> Token putTag(Annotation.Source source, T tag) {
    super.putTag(source, tag);
    return this;
  }
  
  /**
   * @return a deep copy of this Token
   */
  public Token clone() {
    return clone(Function.identity());
  }
  
  public Token clone(Function<String,String> transform) {
    String text = transform.apply(getText());
    Token result = new Token(text, getBegin(), getEnd());
    // FIXME: tags
    // FIXME: vectors
    //putTag(Annotation.Source.GOLD, getTag(Annotation.Source.GOLD, BIO2Tag.class));
    return result;
  }

  @Override
  public boolean equals(Object o) {
    if(this == o) {
      return true;
    }
    if(!(o instanceof Token)) {
      return false;
    }
    if(!super.equals(o)) {
      return false;
    }
    Token token = (Token) o;
    return super.equals(token) &&
           Objects.equals(getText(), token.getText());
  }

  @Override
  public int hashCode() {
    return Objects.hash(super.hashCode(), getText());
  }
}
