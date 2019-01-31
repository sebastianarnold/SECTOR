package de.datexis.annotator;

import de.datexis.common.Configuration;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Set;
import javax.xml.bind.annotation.XmlRootElement;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Provenance information that can be used for result export
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
@XmlRootElement
public class Provenance {

  protected final static Logger log = LoggerFactory.getLogger(Provenance.class);
  private static SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMdd");
  
  /** Annotator Name, e.g. MentionAnnotator */
  protected String name;
  
  /** Model language, e.g. en, de */
  protected String language;
  
  /** Model task, e.g. NER-GENERIC */
  protected String task;
  
  /** Training dataset, e.g. WikiNER */
  protected String dataset;
  
  /** Network architecture, e.g. BLSTM-TRI-POS-EMB */
  protected String architecture;
  
  /** Feature distinction, e.g. +emb+tri */
  protected String features;
  
  /** Model date, e.g. 20170309 */
  protected Date date;
  
  /** Annotator system, e.g. de.datexis.texoo */
  protected String annotator;
  
  /** Version, e.g. 0.5-stable */
  protected String version;
  
  /** Annotator GIT commit, e.g. 55947827b360fc0c92e7a3d31b7ccf953e2f5ae0 */
  protected String commit;

  protected Provenance() { }
  
  public Provenance(Class parent) {
    name = parent.getSimpleName();
    annotator = "de.datexis.texoo";
    commit = "HEAD";
    date = new Date();
    version = Configuration.getVersion();
  }
  
  public String getName() {
    return name;
  }

  public Provenance setName(String name) {
    this.name = name;
    return this;
  }

  public String getAnnotator() {
    return annotator;
  }

  public Provenance setAnnotator(String annotator) {
    this.annotator = annotator;
    return this;
  }

  public String getVersion() {
    return version;
  }

  public Provenance setVersion(String version) {
    this.version = version;
    return this;
  }
  
  public String getCommit() {
    return commit;
  }

  public Provenance setCommit(String commit) {
    this.commit = commit;
    return this;
  }

  public String getDate() {
    if(date == null) return "";
    else return dateFormat.format(date);
  }

  public Provenance setDate(String date) {
    try {
      this.date = dateFormat.parse(date);
    } catch (ParseException ex) {
      this.date = null;
    }
    return this;
  }
  
  public Provenance setDate(Date date) {
    this.date = date;
    return this;
  }

  public String getTask() {
    return task;
  }

  public Provenance setTask(String task) {
    this.task = task;
    return this;
  }

  public String getLanguage() {
    return language;
  }

  public Provenance setLanguage(String language) {
    if(language != null) this.language = language.toLowerCase();
    return this;
  }

  public String getDataset() {
    return dataset;
  }

  public Provenance setDataset(String dataset) {
    this.dataset = dataset;
    return this;
  }

  public String getArchitecture() {
    return architecture;
  }

  public Provenance setArchitecture(String architecture) {
    this.architecture = architecture;
    return this;
  }
  
  void setArchitecture(String id, Set<String> encoders) {
    StringBuilder out = new StringBuilder();
    if(!id.isEmpty()) out.append("_").append(id);
    for(String e : encoders) {
      out.append("_").append(e);
    }
    setArchitecture(out.toString().substring(1));
  }

  public String getFeatures() {
    return features;
  }

  public Provenance setFeatures(String featuresShort) {
    if(!featuresShort.startsWith("+")) featuresShort = "+" + featuresShort.replaceAll("[,\\s]", "+");
    this.features = featuresShort;
    return this;
  }
  
  @Override
  public String toString() {
    StringBuilder out = new StringBuilder();
    if(getName() != null)     out.append("_").append(getName());
    if(getLanguage() != null) out.append("_").append(getLanguage());
    if(getTask()!= null)      out.append("_").append(getTask());
    if(getDataset()!= null)   out.append("_").append(getDataset());
    if(getFeatures()!= null)  out.append(getFeatures()); // starts with +
    if(getDate()!= null)      out.append("_").append(getDate());
    return out.substring(1);
  }

}
