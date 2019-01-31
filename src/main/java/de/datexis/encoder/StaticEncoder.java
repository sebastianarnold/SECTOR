package de.datexis.encoder;

import de.datexis.common.Resource;
import de.datexis.model.Document;
import java.util.Collection;

/**
 * An encoder that does not need to be trained.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public abstract class StaticEncoder extends Encoder {

  public StaticEncoder(String id) {
    super(id);
    this.modelAvailable = true;
  }
  
  @Override
  public void loadModel(Resource modelFile) {
    throw new UnsupportedOperationException("Not applicable to StaticEncoder.");
  }

  @Override
  public void saveModel(Resource modelPath, String name) {
    throw new UnsupportedOperationException("Not applicable to StaticEncoder.");
  }
  
  @Override
  public void trainModel(Collection<Document> documents) {
    log.debug("Static Encoder does not need to be trained.");
  }
  
}
