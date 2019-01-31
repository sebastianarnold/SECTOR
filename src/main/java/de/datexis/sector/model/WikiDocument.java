package de.datexis.sector.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import de.datexis.model.Document;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
@JsonPropertyOrder({ "id", "type", "title", "abstract", "text", "annotations" })
@JsonIgnoreProperties({"begin", "length", "language"})
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY, property = "class", defaultImpl=WikiDocument.class)
public class WikiDocument extends Document {

  protected final static Logger log = LoggerFactory.getLogger(WikiDocument.class);

  protected String title;  
  protected String abstr;

  public String getTitle() {
    return title;
  }

  public void setTitle(String title) {
    this.title = title;
  }
  
  public String getAbstract() {
    return abstr;
  }

  public void setAbstract(String abstr) {
    this.abstr = abstr;
  }

}
