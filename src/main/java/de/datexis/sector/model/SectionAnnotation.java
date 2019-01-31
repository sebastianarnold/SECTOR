package de.datexis.sector.model;

import com.fasterxml.jackson.annotation.*;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.sector.encoder.ClassTag;

import java.util.stream.Collectors;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An Annotation used to attach a phrase (e.g. sentences) to documents, e.g. questions, summarizations, etc.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
@JsonPropertyOrder({"class", "source", "begin", "length", "sectionHeading", "sectionLabel" })
@JsonIgnoreProperties({"confidence", "text"})
public class SectionAnnotation extends Annotation {

  protected final static Logger log = LoggerFactory.getLogger(SectionAnnotation.class);

  protected String type;
  protected String sectionHeading;
  protected String sectionLabel;
  
  /**  Used for JSON Deserialization */
  protected SectionAnnotation() {}
  
  public SectionAnnotation(Annotation.Source source) {
    super(source, "");
  }
  
  public SectionAnnotation(Annotation.Source source, String type, String sectionHeading) {
    super(source, "");
    this.type = type;
    this.sectionHeading = sectionHeading;
  }

  public String getSectionLabel() {
    return sectionLabel;
  }
  
  public void setSectionLabel(String label) {
    this.sectionLabel = label;
  }
  
  public String getSectionHeading() {
    return sectionHeading;
  }

  public void setSectionHeading(String sectionHeading) {
    this.sectionHeading = sectionHeading;
  }
  
  @JsonIgnore
  public String getSectionLabelOrHeading() {
    if(sectionLabel != null && !sectionLabel.isEmpty()) return sectionLabel;
    if(sectionHeading != null && !sectionHeading.isEmpty()) return sectionHeading;
    else return "";
  }
  
  @JsonIgnore
  public String getType() {
    return type;
  }
 
  /**
   * Returns TRUE, if this Annotation matches the boundaries and class of another annotation
   * @param other
   * @return 
   */
  public boolean matches(SectionAnnotation other) {
    if(!this.getSectionLabel().equals(other.getSectionLabel())) return false;
    int p1 = this.getBegin();
    int p2 = other.getBegin();
    int e1 = p1 + this.getLength() - 1;
    int e2 = p2 + other.getLength() - 1;
    return (p1<=p2 && p2<=e1) || (p1<=e2 && e2<=e1) ||
           (p2<=p1 && p1<=e2) || (p2<=e1 && e1<=e2);
  }
  
  /**
   * Two annotations are equal, iff begin and length are equal
   * @param obj
   * @return 
   */
  @Override
  public boolean equals(Object obj) {
    if(this == obj) return true;
    if(obj == null) return false;
    if(getClass() != obj.getClass()) return false;
    final SectionAnnotation other = (SectionAnnotation) obj;
    if(this.begin != other.getBegin()) return false;
    if(this.end != other.getEnd()) return false;
    // if(this.source != other.source) return false; // DON'T include source to compare GOLD and PRED
    if(sectionLabel == null) {
			if(other.getSectionLabel() != null) return false;
		} else if(!sectionLabel.equals(other.getSectionLabel())) return false;
    return true;
  }

}
