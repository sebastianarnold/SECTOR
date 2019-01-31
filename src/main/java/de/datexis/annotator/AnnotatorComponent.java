package de.datexis.annotator;

import de.datexis.tagger.Tagger;
import de.datexis.common.Resource;
import de.datexis.common.Timer;
import de.datexis.encoder.Encoder;
import de.datexis.encoder.EncoderSet;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.StreamSupport;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.shade.jackson.annotation.JsonIgnore; // it is import to use the nd4j version in this class!
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonInclude.Include;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Superclass for Components in an Annotator
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
@JsonInclude(Include.NON_NULL)
@JsonIgnoreProperties(ignoreUnknown = true)
public abstract class AnnotatorComponent {
    
  protected Logger log = LoggerFactory.getLogger(AnnotatorComponent.class);

  private String name;
  private String model = null;

  protected String id;
  
  protected Timer timer = new Timer();
  private StringBuilder trainLog = new StringBuilder();
  private StringBuilder testLog = new StringBuilder();
  
  // TODO: howto model EncoderSet?
  private final List<Tagger> taggers = new ArrayList<>();
  protected EncoderSet encoders = new EncoderSet();
  protected EncoderSet targetEncoders = new EncoderSet();
  
  protected boolean modelAvailable = false;

  public AnnotatorComponent(boolean modelAvailable) {
    this.modelAvailable = modelAvailable;
  }
  
  /**
	 * @return The name of the model
	 */
  public String getName() {
    return name;
  }

  public void setName(String name) {
    this.name = name;
  }

  public String getId() {
    return id;
  }

  public void setId(String id) {
    this.id = id;
  }
  
  public void setModelAvailable(boolean modelAvailable) {
    this.modelAvailable = modelAvailable;
  }
  
  /**
   * @return True, iff all models in this Component (including children) are loaded and trained.
   */
  @JsonIgnore
  public boolean isModelAvailable() {
    return modelAvailable && isModelAvailableInChildren();
  }
  
  /**
   * @return True, iff all models in all children components are loaded and trained.
   */
  @JsonIgnore
  public boolean isModelAvailableInChildren() {
    return StreamSupport.stream(encoders.spliterator(), false).allMatch(child -> child.isModelAvailable())
            && taggers.stream().allMatch(child -> child.isModelAvailable());
  }
  
  /**
   * @return JSON representation of Component configuration
   */
  @JsonIgnore
  public String getConf() {
    try {
      //NeuralNetConfiguration.mapper().setSerializationInclusion(JsonInclude.Include.NON_NULL);
      String json = NeuralNetConfiguration.mapper().writer().writeValueAsString(this);
      return json.replaceAll("\\s", "");
    } catch (JsonProcessingException ex) {
      log.error("Could not serialize class to JSON: " + ex.toString());
      return null;
    }
  }

  /**
   * Initializes the Component and sets configuration
   */
  public void setConf() {
    throw new UnsupportedOperationException();
  }
  /**
   * @return Model reference as String (file reference or URL)
   */
  @JsonIgnore
  public String getModel() {
    return model == null ? "" : model;
  }

  /**
   * Sets the model Resource. Called by loadModel or saveModel.
   * @param model 
   */
  protected void setModel(Resource model) {
    if(model == null) this.model = null;
    else this.model = model.getFileName();
  }
  
  protected void setModelFilename(String model) {
    this.model = model;
  }
  
  @Deprecated // use addInputEncoder
  public void addEncoder(Encoder e) {
    encoders.addEncoder(e);
  }
  
  public void addInputEncoder(Encoder e) {
    encoders.addEncoder(e);
  }

  public void addTargetEncoder(Encoder e) {
    targetEncoders.addEncoder(e);
  }
  
  public AnnotatorComponent setEncoders(EncoderSet encs) {
    encoders = encs;
    return this;
  }
  
  @JsonIgnore
  public EncoderSet getEncoders() {
    return encoders;
  }
  
  @JsonIgnore
  public EncoderSet getTargetEncoders() {
    return targetEncoders;
  }
  
  @JsonIgnore
  public Iterable<? extends AnnotatorComponent> getChildren() {
    return taggers;
  }
  
  /**
	 * Load a pre-trained model
   * @param file The file to load
	 */
  public abstract void loadModel(Resource file) throws IOException;
  
  /**
	 * Load a pre-trained model
   * @param dir The path to create the file
   * @param name The name of the model. File extension will be added automatically.
	 */
  public abstract void saveModel(Resource dir, String name);
    
  public void appendTrainLog(String message) {
    trainLog.append(message).append("\n");
    log.info(message);
  }
  
  public void appendTrainLog(String message, long time) {
    String msg = message + " [" + Timer.millisToLongDHMS(time) + "]";
    trainLog.append(msg).append("\n");
    log.info(msg);
  }
  
  public void appendTestLog(String message) {
    testLog.append(message).append("\n");
    //log.info(message);
  }
  
  public void appendTestLog(String message, long time) {
    String msg = message + " [" + Timer.millisToLongDHMS(time) + "]";
    testLog.append(msg).append("\n");
    log.info(msg);
  }
  
  protected String getTrainLog() {
    return trainLog.toString();
  }
  
  protected String getTestLog() {
    return testLog.toString();
  }

  protected void clearTrainLog() {
    trainLog = new StringBuilder();
  }
  
  protected void clearTestLog() {
    testLog = new StringBuilder();
  }
 
}