package de.datexis.tagger;

import de.datexis.common.Resource;
import de.datexis.model.Document;
import java.util.Collection;

/**
 * An empty Tagger that is used to initialize an Annotator from scratch.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class EmptyTagger extends Tagger {
  public EmptyTagger() {
    super(0, 0);
    setId("NULL");
    setModelAvailable(true);
  }

  @Override
  public void saveModel(Resource modelPath, String name) {
  }
  
  @Override
  public void loadModel(Resource modelFile) {
  }

  @Override
  public void tag(Collection<Document> docs) {
  }
}