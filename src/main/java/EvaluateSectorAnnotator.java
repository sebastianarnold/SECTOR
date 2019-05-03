import de.datexis.annotator.AnnotatorFactory;
import de.datexis.common.*;
import de.datexis.model.*;
import de.datexis.sector.SectorAnnotator;
import de.datexis.sector.reader.WikiSectionReader;
import java.io.IOException;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Run experiments on a pre-trained SECTOR model
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class EvaluateSectorAnnotator {
  
  protected final static Logger log = LoggerFactory.getLogger(EvaluateSectorAnnotator.class);
  
  public static void main(String[] args) throws IOException {
    
    final EvaluateSectorAnnotator.ExecParams params = new EvaluateSectorAnnotator.ExecParams();
    final CommandLineParser parser = new CommandLineParser(params);
    
    try {
      parser.parse(args);
      new EvaluateSectorAnnotator().evaluate(params);
      System.exit(0);
    } catch(ParseException e) {
      HelpFormatter formatter = new HelpFormatter();
      formatter.printHelp("sector-eval", "SECTOR: evaluate SectorAnnotator on WikiSection dataset", params.setUpCliOptions(), "", true);
      System.exit(1);
    } catch(Exception e) {
      e.printStackTrace();
      System.exit(1);
    }
    
  }
  
  protected static class ExecParams implements CommandLineParser.Options {
    
    protected String modelPath = null;
    protected String testFile = null;
    protected String embeddingsPath = null;
    
    @Override
    public void setParams(CommandLine parse) {
      modelPath = parse.getOptionValue("m");
      testFile = parse.getOptionValue("t");
      embeddingsPath = parse.getOptionValue("e");
    }
    
    @Override
    public Options setUpCliOptions() {
      Options op = new Options();
      op.addRequiredOption("m", "model", true, "path to the pre-trained model");
      op.addRequiredOption("t", "test", true, "file name of WikiSection test dataset");
      op.addOption("e", "embedding", true, "search path to word embedding models (if not provided by the model itself)");
      return op;
    }
    
  }
  
  public void evaluate(EvaluateSectorAnnotator.ExecParams params) throws IOException {
    
    // Configure parameters
    Resource testFile = Resource.fromDirectory(params.testFile);
    Resource modelPath = Resource.fromDirectory(params.modelPath);
    Resource embeddingsPath = params.embeddingsPath != null ? Resource.fromDirectory(params.embeddingsPath) : null;
    
    // Load model
    SectorAnnotator sector = (SectorAnnotator) (params.embeddingsPath != null ?
      AnnotatorFactory.loadAnnotator(modelPath, embeddingsPath) :
      AnnotatorFactory.loadAnnotator(modelPath));
  
    // Read dataset
    Dataset test = WikiSectionReader.readDatasetFromJSON(testFile);
    
    // Annotate documents
    //sector.getTagger().setBatchSize(8); // if you need to save RAM on CUDA device
    // will attach SectorEncoder vectors to Sentences and create SectionAnnotations
    sector.annotate(test.getDocuments(), SectorAnnotator.SegmentationMethod.BEMD);
    
    // Evaluate annotated documents for segmentation and segment classification
    sector.evaluateModel(test, false, true, true);
    
  }
  
}
