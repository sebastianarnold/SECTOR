package de.datexis.annotator;

import de.datexis.tagger.Tagger;
import de.datexis.common.Resource;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.tagger.AbstractIterator;
import de.datexis.tagger.EmptyTagger;
import de.datexis.preprocess.DocumentFactory;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Collection;
import java.util.Map;
import java.util.TreeMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An Annotator runs a configuration of Components on a Document or Dataset.
 * A root Tagger is used to attach Annotations to the Document.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class Annotator {

  protected final static Logger log = LoggerFactory.getLogger(Annotator.class);
  
  /**
   * The root tagger that is called for this Annotator.
   */
  protected Tagger tagger = new EmptyTagger();
  
  /**
   * The AbstractIterator used to call the root Tagger.
   */
  protected AbstractIterator it;
  
  /**
   * A Map of <ID,Component> for all Components used by this Annotator.
   * Each Component knows its children after the model was loaded.
   */
  protected Map<String,AnnotatorComponent> components = new TreeMap<>();
  
  /**
   * Provenance information, e.g. Version, GIT commit, training data, ...
   */
  protected Provenance provenance;
  
  /**
   * Used by AnnotatorFactory
   */
  protected Annotator() {
    provenance = new Provenance(this.getClass());
  }
  
  public Annotator(Tagger root) {
    provenance = new Provenance(this.getClass());
    this.tagger = root;
  }
  
  public Annotator(AnnotatorComponent component) {
    provenance = new Provenance(this.getClass());
    addComponent(component);
  }

  public Document annotate(String text) {
    Document doc = DocumentFactory.fromText(text);
    annotate(doc);
    return doc;
  }
  
  public Document annotate(Document doc) { 
    annotate(Arrays.asList(doc));
    return doc;
  }
  
  public Dataset annotate(Dataset data) {
    annotate(data.getDocuments());
    return data;
  }
  
  public void annotate(Collection<Document> docs) {
    tagger.tag(docs);
  }
  
  public Document createDocument(String text) {
    return DocumentFactory.fromText(text);
  }
  
  public Dataset createDataset(String text) {
    Dataset data = new Dataset();
    data.addDocument(createDocument(text));
    return data;
  }
  
  public void addComponent(AnnotatorComponent component) {
    String id = component.getId();
    int inc = 0;
    while(components.containsKey(id)) {
      log.warn("Component with id " + id + " already exists in Annotator, incrementing");
      id = component.getId() + inc++;
      component.setId(id);
    }
    components.put(id, component);
    if(tagger != null) provenance.setArchitecture(tagger.getId(), components.keySet());
    else provenance.setArchitecture("", components.keySet());
  }
  
  public Tagger getTagger() {
    return tagger;
  }

  public Provenance getProvenance() {
    return provenance;
  }

  /**
   * Writes annotator.xml and binary model
   * @param path Directory to write to
   */
  public void writeModel(Resource path) {
    writeModel(path, "annotator");
  }
  
  /**
   * Writes <name>.xml and binary models
   * @param path Directory to write to
   */
  public void writeModel(Resource path, String name) {
    log.info("Writing model to {}", path.toString());
    writeComponents(path);
    AnnotatorFactory.writeXML(this, path.resolve(name + ".xml"));
  }
  
  /**
   * Reads and initializes a model (can be partly untrained)
   * @param file ZIP file or directory that includes annotator.xml
   */
  public void readModel(Resource file) {
    throw new UnsupportedOperationException();
  }
  
  public void writeComponents(Resource path) {
      if(tagger != null) tagger.saveModel(path, tagger.getId().toLowerCase());
      for(AnnotatorComponent comp : components.values()) {
        try {
          comp.saveModel(path, comp.getId().toLowerCase());
        } catch(UnsupportedOperationException ex) {
          // some models do not need to save data
        }
      }
  }
  
  public boolean isModelAvailable() {
    return tagger.isModelAvailable();
  }
  
  public boolean isModelAvailableInChildren() {
    return tagger.isModelAvailableInChildren();
  }
  
  public AnnotatorComponent getComponent(String id) {
    return components.get(id);
  }
  
  /**
   * Trains all Components that are marked for training
   */
  public void trainModel(Dataset train) {
    provenance.setDataset(train.getName());
    provenance.setLanguage(train.getLanguage());
    tagger.trainModel(train);
  }
  
  /**
   * Writes train.log
   * @param path directory to write to
   */
  public void writeTrainLog(Resource path) {
    try(PrintStream out = new PrintStream(path.resolve("train.log").getOutputStream(true))) {
      out.print("==== TRAIN: " + tagger.getName() + " =====\n");
      out.print(tagger.getTrainLog());
      out.flush();
    } catch (IOException ex) {
      log.error("Could not write train.log to path '" + path.toString() + "'");
    }
  }
  
  /**
   * Writes eval.log
   */
  public void writeTestLog(Resource path) {
     try(PrintStream out = new PrintStream(path.resolve("test.log").getOutputStream(true))) {
      out.print("==== TEST: " + tagger.getName() + " =====\n");
      out.print(tagger.getTestLog());
      out.flush();
    } catch (IOException ex) {
      log.error("Could not write test.log to path '" + path.toString() + "'");
    }
  }
  
  public void writeHTML() {
    // TODO: implement
    throw new UnsupportedOperationException("note refactored yet");
  }

}
