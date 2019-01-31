package de.datexis.tagger;

import de.datexis.annotator.AnnotatorComponent;
import de.datexis.common.Resource;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Collection;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.nd4j.shade.jackson.databind.JsonNode;

/**
 * A Tagger is a neural network that adds Tags to Spans, e.g. LSTM for Named Entity Recognition.
 * @author sarnold
 */
public abstract class Tagger extends AnnotatorComponent {
  
  protected static final Logger log = LoggerFactory.getLogger(Tagger.class);

  /**
   * Size of the input vector
   */
  protected long inputVectorSize;
  
  /**
   * Size of the output vector
   */
  protected long outputVectorSize;
  
  /**
   * The network to train
   */
  protected Model net;
  
  public Tagger(String id) {
    super(false);
    this.id = id;
  }
  
  protected Tagger(long inputVectorSize, long outputVectorSize) {
    super(false);
    this.inputVectorSize = inputVectorSize; // number of inputs (I)
    this.outputVectorSize = outputVectorSize; // number of outputs (K)
  }
  
  protected Tagger(Resource modelFile) {
    super(false);
    loadModel(modelFile);
    //int i = lstm.getnLayers();
    // FIXME: vector sizes!
    //this.inputVectorSize = lstm.getLayerWiseConfigurations().getConf(0).
    //this.outputVectorSize = outputVectorSize; // number of outputs (K)
  }
  
  public void tag(Stream<Document> docs) {
    tag(docs.collect(Collectors.toList()));
  }
  
  public abstract void tag(Collection<Document> docs);
  
  public void trainModel(Dataset train) {
    trainModel(train, Annotation.Source.GOLD);
  }
  
  public void trainModel(Dataset train, Annotation.Source expected) {
    throw new UnsupportedOperationException("Training not implemented");
  }
  
  /*public void trainModel(Stream<Document> train) {
    throw new UnsupportedOperationException("Training not implemented");
  }*/
  
  public void testModel(Dataset test) {
    testModel(test, Annotation.Source.GOLD);
  }
  
  public void testModel(Dataset test, Annotation.Source expected) {
    throw new UnsupportedOperationException("Testing not implemented");
  }
  
  @JsonIgnore
  public Model getNN() {
    return net;
  }
  
  public ComputationGraphConfiguration getGraphConfiguration() {
    if(net == null) return null;
    else if(net instanceof ComputationGraph) return ((ComputationGraph) net).getConfiguration();
    else return null;
  }
  
  public void setGraphConfiguration(JsonNode conf) {
    if(conf != null) {
      String json = conf.toString();
      if(json != null && !json.equals("null")) {
        net = new ComputationGraph(ComputationGraphConfiguration.fromJson(json));
        net.init();
      }
    }
  }
  
  public MultiLayerConfiguration getLayerConfiguration() {
    if(net == null) return null;
    else if(net instanceof MultiLayerNetwork) return ((MultiLayerNetwork) net).getLayerWiseConfigurations();
    else return null;
  }
  
  public void setLayerConfiguration(JsonNode conf) {
    if(conf != null) {
      String json = conf.toString();
      if(json != null && !json.equals("null")) {
        net = new MultiLayerNetwork(MultiLayerConfiguration.fromJson(json));
        net.init();
      }
    }
  }
  
  public void setListeners(IterationListener... listeners) {
    net.setListeners(listeners);
  }
  
  /**
   * Saves the model to <name>.bin.gz
   * @param modelPath
   * @param name 
   */
  @Override
  public void saveModel(Resource modelPath, String name) {
    if(net instanceof ComputationGraph) {
      Resource modelFile = modelPath.resolve(name + ".zip");
      try(OutputStream os = modelFile.getOutputStream()){
        ModelSerializer.writeModel(net, os, true);
        setModel(modelFile);
      } catch (IOException ex) {
        log.error(ex.toString());
      } 
    } else if(net instanceof MultiLayerNetwork) {
      Resource modelFile = modelPath.resolve(name + ".bin.gz");
      try(DataOutputStream dos = new DataOutputStream(modelFile.getGZIPOutputStream())){
        // Write the network parameters:
        Nd4j.write(net.params(), dos);
        dos.flush();
        setModel(modelFile);
      } catch (IOException ex) {
        log.error(ex.toString());
      } 
    }
  }
  
  @Override
  public void loadModel(Resource modelFile) {
    if(modelFile.getFileName().endsWith("zip")) {
      try(InputStream is = modelFile.getInputStream()) {
        net = ModelSerializer.restoreComputationGraph(is, true);
        setModel(modelFile);
        setModelAvailable(true);
        log.info("loaded ComputationGraph from " + modelFile.getFileName());
      } catch (IOException ex) {
        log.error(ex.toString());
      }
    } else {
      try(DataInputStream dis = new DataInputStream(modelFile.getInputStream())) {
        INDArray newParams = Nd4j.read(dis);
        ((MultiLayerNetwork)net).setParameters(newParams);
        setModel(modelFile);
        setModelAvailable(true);
        log.info("loaded MultiLayerNetwork from " + modelFile.getFileName());
      } catch (IOException ex) {
        log.error(ex.toString());
      }
    }
  }
  
  @Deprecated
  public void saveUpdater(Resource modelPath, String name) {
    Resource modelFile = modelPath.resolve(name + ".bin.gz");
    INDArray updaterState = null;
    if(net instanceof MultiLayerNetwork) updaterState = ((MultiLayerNetwork) net).getUpdater().getStateViewArray();
    else if(net instanceof ComputationGraph) updaterState = ((ComputationGraph) net).getUpdater().getStateViewArray();
    if(updaterState != null) try(DataOutputStream dos = new DataOutputStream(modelFile.getGZIPOutputStream())){
      Nd4j.write(updaterState, dos);
      dos.flush();
    } catch (IOException ex) {
      log.error(ex.toString());
    } 
  }
  
  public void loadConf(Resource confFile) {
    try {
      // Load network configuration from disk:
      MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(IOUtils.toString(confFile.getInputStream()));
      // Create a MultiLayerNetwork from the saved configuration and parameters
      //confFromJson.setTrainingWorkspaceMode(WorkspaceMode.SINGLE);
      //confFromJson.setInferenceWorkspaceMode(WorkspaceMode.SINGLE);
      net = new MultiLayerNetwork(confFromJson);
      net.init();
    } catch (IOException ex) {
      log.error(ex.toString());
    }
  }
  
  public void printModelStats() {
    /*Layer[] layers = net.getLayers();
		int totalNumParams = 0;
		for (int i = 0; i < layers.length; i++) {
			int nParams = layers[i].numParams();
			log.debug("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		log.debug("Total number of network parameters: " + totalNumParams);*/
  }
  
}
