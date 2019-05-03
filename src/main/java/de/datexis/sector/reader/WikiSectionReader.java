package de.datexis.sector.reader;

import de.datexis.sector.model.WikiDocument;
import de.datexis.common.ObjectSerializer;
import de.datexis.common.Resource;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.reader.DatasetReader;
import de.datexis.sector.model.SectionAnnotation;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Reads WikiSection Datasets from JSON file.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class WikiSectionReader implements DatasetReader {

  protected final static Logger log = LoggerFactory.getLogger(WikiSectionReader.class);
  
  @Override
  public Dataset read(Resource path) throws IOException {
    return readDatasetFromJSON(path);
  }
  
  public static Dataset readDatasetFromJSON(Resource path) throws IOException {
    log.info("Reading Wiki Articles from {}", path.toString());
    ObjectSerializer.getObjectMapper().registerSubtypes(SectionAnnotation.class);
    Dataset result = new Dataset(path.getFileName().replace(".json", ""));
    Iterator<Document> it = ObjectSerializer.readJSONDocumentIterable(path);
    while(it.hasNext()) {
      Document doc = it.next();
      for(Annotation ann : doc.getAnnotations()) {
        ann.setSource(Annotation.Source.GOLD);
        ann.setConfidence(1.0);
      }
      if(!doc.isEmpty()) result.addDocument(doc);
      else log.warn("read empty document {}", doc.getId());
    }
    return result;
  }
  
  public static List<WikiDocument> readWikiDocumentsFromJSON(Resource path) throws IOException {
    log.info("Reading Wiki Articles from {}", path.toString());
    ObjectSerializer.getObjectMapper().registerSubtypes(SectionAnnotation.class);
    List<WikiDocument> result = new ArrayList<>();
    Iterator<WikiDocument> it = ObjectSerializer.getObjectMapper().readerFor(WikiDocument.class).readValues(path.getInputStream());
    while(it.hasNext()) {
      WikiDocument doc = it.next();
      for(Annotation ann : doc.getAnnotations()) {
        ann.setSource(Annotation.Source.GOLD);
      }
      result.add(doc);
    }
    return result;
  }

}
