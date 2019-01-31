import de.datexis.common.*;
import de.datexis.common.CommandLineParser;
import de.datexis.encoder.impl.*;
import de.datexis.model.*;
import de.datexis.sector.SectorAnnotator;
import de.datexis.sector.encoder.*;
import de.datexis.sector.model.SectionAnnotation;
import de.datexis.sector.reader.WikiSectionReader;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.cli.*;
import org.deeplearning4j.ui.api.UIServer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Main Controller for training of SECTOR models.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class TrainSectorAnnotator {

  protected final static Logger log = LoggerFactory.getLogger(TrainSectorAnnotator.class);

  public static void main(String[] args) throws IOException {
    
    final ExecParams params = new ExecParams();
    final CommandLineParser parser = new CommandLineParser(params);
    
    try {
      parser.parse(args);
      new TrainSectorAnnotator().runTraining(params);
      System.exit(0);
    } catch(ParseException e) {
      HelpFormatter formatter = new HelpFormatter();
      formatter.printHelp("sector-train", "SECTOR: train SectorAnnotator from WikiSection dataset", params.setUpCliOptions(), "", true);
      System.exit(1);
    } catch(Exception e) {
      e.printStackTrace();
      System.exit(1);
    }
   
  }
  
  protected static class ExecParams implements CommandLineParser.Options {

    protected String trainFile;
    protected String devFile = null;
    protected String testFile = null;
    protected String outputPath = null;
    protected String embeddingsFile = null;
    protected String language = null;
    protected boolean trainingUI = false;
    protected boolean isHeadingsModel = false;

    @Override
    public void setParams(CommandLine parse) {
      trainFile = parse.getOptionValue("i");
      devFile = parse.getOptionValue("v");
      testFile = parse.getOptionValue("t");
      outputPath = parse.getOptionValue("o");
      embeddingsFile = parse.getOptionValue("e");
      language = parse.getOptionValue("l", "en");
      trainingUI = parse.hasOption("u");
      isHeadingsModel = parse.hasOption("h");
    }

    @Override
    public Options setUpCliOptions() {
      Options op = new Options();
      op.addRequiredOption("i", "input", true, "file name of WikiSection training dataset");
      op.addRequiredOption("o", "output", true, "path to create and store the model");
      op.addOption("h", "headings", false, "train multi-label model (SEC>H), otherwise single-label model (SEC>T) is used");
      op.addRequiredOption("o", "output", true, "path to create and store the model");
      op.addOption("v", "validation", true, "file name of WikiSection validation dataset (will use early stopping if given)");
      op.addOption("t", "test", true, "file name of WikiSection test dataset (will test after training if given)");
      op.addOption("e", "embedding", true, "path to word embedding model, will use bloom filters if not given");
      op.addOption("l", "language", true, "language to use for sentence splitting and stopwords (EN or DE)");
      op.addOption("u", "ui", false, "enable training UI (http://127.0.0.1:9000)");
      return op;
    }

  }
  
  protected void runTraining(ExecParams params) throws IOException {
    
    // Configure parameters
    Resource trainingPath = Resource.fromDirectory(params.trainFile);
    Resource validationPath = params.devFile != null ? Resource.fromDirectory(params.devFile) : null;
    Resource testPath = params.testFile != null ? Resource.fromDirectory(params.testFile) : null;
    Resource output = Resource.fromDirectory(params.outputPath);
    WordHelpers.Language lang = WordHelpers.getLanguage(params.language);

    // Read datasets
    Dataset train = trainingPath.getFileName().endsWith(".json") ?
      WikiSectionReader.readDatasetFromJSON(trainingPath) :
      WikiSectionReader.readDatasetFromJSON(trainingPath);
    Dataset validation = validationPath == null ? null :
      validationPath.getFileName().endsWith(".json") ?
      WikiSectionReader.readDatasetFromJSON(validationPath) :
      WikiSectionReader.readDatasetFromJSON(validationPath);
    Dataset test = testPath == null ? null :
      testPath.getFileName().endsWith(".json") ?
        WikiSectionReader.readDatasetFromJSON(testPath) :
        WikiSectionReader.readDatasetFromJSON(testPath);

    SectorAnnotator.Builder builder = new SectorAnnotator.Builder();

    // Configure input encoders (bloom filter or word embeddings)
    if(params.embeddingsFile == null) initializeInputEncodings_bloom(builder, train, lang);
    else initializeInputEncodings_wemb(builder, Resource.fromFile(params.embeddingsFile));

    // Configure target encoders (class labels or heading labels)
    if(params.isHeadingsModel) initializeHeadingsTarget(builder, train, lang);
    else initializeClassLabelsTarget(builder, train);

    // Build the Annotator
    SectorAnnotator sector = builder
      .withDataset(train.getName(), lang)
      .withModelParams(0, 256, 128)                     // ffwLayerSize, lstmLayerSize, embeddingLayerSize is hardcoded here
      .withTrainingParams(0.01, 0.5, 2048, 396, 16, 10) // learningrate, dropout, epochsize, maxlength, batchsize, epochs is hardcoded here
      .enableTrainingUI(params.trainingUI)
      .build();

    boolean success = false;
    try {

      // Train model
      if(validation == null) sector.trainModel(train);
      else sector.trainModelEarlyStopping(train, validation, 10, 10, 100); // minepochs, tryepochs, maxepochs is hardcoded here

      // Save model
      output = output.resolve(sector.getTagger().getName());
      output.toFile().mkdirs();
      sector.writeModel(output);
      sector.writeTrainLog(output);

      // Test model
      if(test != null) {
        sector.annotate(test.getDocuments(), SectorAnnotator.SegmentationMethod.BEMD);
        sector.evaluateModel(test, false, true, true);
      }
      sector.writeTestLog(output);
      success = true;

    } finally {
      try {
        // try to stop the training UI without exception handling and then exit
        if(params.trainingUI) UIServer.getInstance().stop();
      } catch(NoClassDefFoundError e) {
      } catch(Exception e) {
      } finally {
        System.exit(success ? 0 : 1);
      }
    }

  }

  protected SectorAnnotator.Builder initializeInputEncodings_bloom(SectorAnnotator.Builder builder, Dataset train, WordHelpers.Language lang) {

    BloomEncoder bloom = new BloomEncoder(4096, 5);
    bloom.trainModel(train.getDocuments(), 5, lang);
    StructureEncoder structure = new StructureEncoder();

    return builder.withInputEncoders("bloom", bloom, new DummyEncoder(), structure);

  }

  protected SectorAnnotator.Builder initializeInputEncodings_wemb(SectorAnnotator.Builder builder, Resource embeddingModel) throws IOException {

    Word2VecEncoder wordEmb = new Word2VecEncoder();
    wordEmb.loadModel(embeddingModel);
    StructureEncoder structure = new StructureEncoder();

    return builder.withInputEncoders("emb", new DummyEncoder(), wordEmb, structure);

  }

  /**
   * Initialize target encoder for the class labels task (single-label classification)
   */
  protected SectorAnnotator.Builder initializeClassLabelsTarget(SectorAnnotator.Builder builder, Dataset train) {
    
    // preprocess Section Annotations
    ArrayList<String> sections = new ArrayList<>();
      for(Document doc : train.getDocuments()) {
        for(SectionAnnotation ann : doc.getAnnotations(SectionAnnotation.class)) {
          sections.add(ann.getSectionLabel());
        }
      }
    
    // build Section Encoder
    ClassEncoder targetEncoder = new ClassEncoder();
    targetEncoder.trainModel(sections, 0);

    return builder
      .withId("SEC>T")
      .withTargetEncoder(targetEncoder)
      .withLossFunction(new LossMCXENT(), Activation.SOFTMAX, false);
    
  }
  
  /**
   * Initialize target encoder for the headings prediction task (multi-label classification)
   */
  protected SectorAnnotator.Builder initializeHeadingsTarget(SectorAnnotator.Builder builder, Dataset train, WordHelpers.Language lang) {
    
    // preprocess Section Annotations
    ArrayList<String> headings = new ArrayList<>();
    for(Document doc : train.getDocuments()) {
      for(SectionAnnotation ann : doc.getAnnotations(SectionAnnotation.class)) {
        headings.add(ann.getSectionHeading());
      }
    }
    
    // build Heading Encoder
    HeadingEncoder targetEncoder = new HeadingEncoder();
    targetEncoder.trainModel(headings, 20, lang); // ignore words with less than 20 occurences

    return builder
      .withId("SEC>H")
      .withTargetEncoder(targetEncoder)
      // for ranking loss, use:
      //.withLossFunction(new MultiClassDosSantosPairwiseRankingLoss(), Activation.SIGMOID, false)
      .withLossFunction(new LossMCXENT(), Activation.SOFTMAX, true);
    
  }
  

  
}
