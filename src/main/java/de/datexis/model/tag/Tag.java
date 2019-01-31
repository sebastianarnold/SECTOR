package de.datexis.model.tag;

import com.fasterxml.jackson.annotation.JsonTypeInfo;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A Tag is used to classify a Token, e.g. BIO2 Label or Mention Label
 * @author sarnold
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY, property = "class")
public interface Tag {
  
  public static final String GENERIC = "GENERIC";
  
  /**
   * @return The size of this Tag's representation vector
   */
  public int getVectorSize();
  
  /**
   * @return The vector that represents this Tag as training target.
   */
  public INDArray getVector();
  
  /**
   * @return A String representation of this Tag.
   */
  public String getTag();
  
  // TODO: refactor these to EnumTag?
  public String getTag(int index);
  
  public double getConfidence();
  
}
