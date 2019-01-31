package de.datexis.reader;

import de.datexis.common.Resource;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import java.io.IOException;
import java.util.stream.Stream;

/**
 * Reads a Dataset from standard format into a TeXoo Dataset.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public interface DatasetReader {
  
  public Dataset read(Resource path) throws IOException;
  
  // TODO: implement for next version
  //public Stream<Document> stream(Resource path);
  
}
