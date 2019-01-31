package de.datexis.encoder.impl;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import de.datexis.encoder.StaticEncoder;
import de.datexis.model.Span;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonIgnore;

/**
 * A an empty dummy Encoder with vector size 1 that returns always [0].
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class DummyEncoder extends StaticEncoder {

  public DummyEncoder() {
    super("dummy");
    setName("dummy");
  }
  
  @Override
  @JsonIgnore
  public long getEmbeddingVectorSize() {
    return 1;
  }

  @Override
  public INDArray encode(Span span) {
    return Nd4j.create(1).transpose();
  }

  @Override
  public INDArray encode(String word) {
    return Nd4j.create(1).transpose();
  }

}
