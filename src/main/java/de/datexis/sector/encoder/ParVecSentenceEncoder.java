package de.datexis.sector.encoder;

import de.datexis.parvec.encoder.ParVecEncoder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Encodes a sentence using ParVec inference.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class ParVecSentenceEncoder extends ParVecEncoder {

  protected final static Logger log = LoggerFactory.getLogger(ParVecSentenceEncoder.class);

  @Override
  public long getEmbeddingVectorSize() {
    // return size of the embedding!
    return layerSize;
  }
  
}
