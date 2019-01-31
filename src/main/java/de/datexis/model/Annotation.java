package de.datexis.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;

/**
 * Marks a Span as Annotated by GOLD standard, SILVER generated, PREDicted or USER generated.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY, property = "class")
public class Annotation extends Span {

  protected final static Logger log = LoggerFactory.getLogger(Annotation.class);

  /**
   * Determines the source of a Annotation: gold standard, predicted from model, user annotated.
   */
  public static enum Source { 
    /** Gold standard. Depicts that an Annotation originates from a gold standard. */
    GOLD, 
    /** Silver generated. Depicts that an Annotation was automatically generated. */
    SILVER,
    /** Predicted. Depicts that a Annotation was predicted by a model. */
    PRED, 
    /** User annotation. Depicts that an Annotation was annotated manually by a user. */
    USER,
    /** Sampled annotation. Depicts that an Annotation was specifically selected for training or testing. */
    SAMPLED,
    /** Training annotation. Depicts that an Annotation was specifically selected for training. */
    TRAIN,
  };
  
  /**
   * Determines if two Annotations match using strong or weak matching algorithms.
   */
  public static enum Match {
    /** Strong annotation match */
    STRONG,
    /** Weak annotation match */
    WEAK,
  };
  
  protected String text;
  protected Source source;
  protected double confidence = 0.;
  
  public Annotation(Source source, String text, int begin, int end) {
    this.source = source;
    this.text = text;
    this.begin = begin;
    this.end = end;
  }
  
  public Annotation(Source source, String text) {
    this(source, text, 0, text.length());
  }
  
  /**
   * Copy constructor
   * @param ann 
   */
  public Annotation(Annotation ann) {
    this(ann.getSource(), ann.getText(), ann.getBegin(), ann.getEnd());
    setDocumentRef(ann.getDocumentRef());
  }

  /**
   * Default constructor.
   * @deprecated only used for JSON deserialization.
   */
  @Deprecated
  protected Annotation() {}
  
  @Override
  public String getText() {
    return text;
  }

  public void setText(String text) {
    this.text = text;
  }

  public Source getSource() {
    return source;
  }

  public void setSource(Source source) {
    this.source = source;
  }

  public double getConfidence() {
    return confidence;
  }

  public void setConfidence(double confidence) {
    this.confidence = confidence;
  }
  
  /**
   * Returns TRUE, if this Annotation shares any character position with another annotation (based on boundaries).
   * @param other
   * @return 
   */
  public boolean intersects(Annotation other) {
    if(this.begin <= other.begin && this.end > other.begin) return true;
    else if(other.begin <= this.begin && other.end > this.begin) return true;
    else return false;
  }
  
  /**
   * Returns TRUE, if the boundaries of this Annotation completely contains another annotation (based on boundaries).
   * @param other
   * @return 
   */
  public boolean contains(Annotation other) {
    if(this.begin <= other.begin && this.end >= other.end) return true;
    else return false;
  }
    
  /**
   * Returns TRUE, if this Annotation matches the boundaries of another annotation
   * @param other
   * @return 
   */
  public boolean matches(Annotation other) {
    return this.matches(other, Match.STRONG);
  }
  
  /**
   * Returns TRUE, if this Annotation matches the boundaries of another annotation.
   * Implemented after Cornolti et al. (2013): A Framework for Benchmarking Entity-Annotation Systems.
   * @param other
   * @param match - WEAK or STRONG boundary match
   * @return 
   */
  public boolean matches(Annotation other, Match match) {
    // if(this.getDocumentRef() != other.getDocumentRef()) return false; // FIXME: deactivated because references are not loaded from JSON
    switch (match) {
      case WEAK:
        int p1 = this.getBegin();
        int p2 = other.getBegin();
        int e1 = p1 + this.getLength() - 1;
        int e2 = p2 + other.getLength() - 1;
        return (p1<=p2 && p2<=e1) || (p1<=e2 && e2<=e1) ||
               (p2<=p1 && p1<=e2) || (p2<=e1 && e1<=e2);
      case STRONG:
      default:
        return this.getBegin() == other.getBegin() &&
               this.getLength() == other.getLength();
    }
  }

  @Override
  public boolean equals(Object o) {
    if(this == o) {
      return true;
    }
    if(!(o instanceof Annotation)) {
      return false;
    }
    if(!super.equals(o)) {
      return false;
    }
    Annotation annotation = (Annotation) o;
    return super.equals(annotation) && 
           Objects.equals(getText(), annotation.getText()) &&
           Objects.equals(getSource(), annotation.getSource());
  }

  @Override
  public int hashCode() {
    return Objects.hash(super.hashCode(), getText(), getSource());
  }
}
