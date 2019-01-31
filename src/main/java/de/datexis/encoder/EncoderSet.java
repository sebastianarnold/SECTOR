package de.datexis.encoder;

import de.datexis.model.Span;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.LoggerFactory;

/**
 * A set of Encoders for vectors that will be concatenated as input
 * @author sarnold
 */
public class EncoderSet implements Iterable<Encoder>, IEncoder {
  
  protected static final org.slf4j.Logger log = LoggerFactory.getLogger(EncoderSet.class);
  
  protected List<Encoder> encoders;
  protected int size;
  
  public EncoderSet(Encoder... encoders) {
    this.encoders = new ArrayList<>(encoders.length);
    this.size = 0;
    for(Encoder enc : encoders) {
      addEncoder(enc);
    }
  }
  
  public final void addEncoder(Encoder e) {
    encoders.add(e);
    if(e.getEmbeddingVectorSize() == 0) log.warn("Adding uninitialized Encoder " + e.getName());
    this.size += e.getEmbeddingVectorSize();
  }
  
  /**
   * Recalculates vector size in case one Encoder has changed
   */
  public void updateVectorSize() {
    this.size = 0;
    for(Encoder enc : this.encoders) {
      this.size += enc.getEmbeddingVectorSize();
    }
  }
  
  @Override
  public long getEmbeddingVectorSize() {
    return size;
  }
  
  public Iterable<Encoder> iterable() {
    return encoders;
  }

  @Override
  public Iterator<Encoder> iterator() {
    return encoders.iterator();
  }
  
  /**
   * Encodes a given String using all Encoders. Does not save the intermediate results to the Tokens.
   */
  public INDArray encode(String word) {
    INDArray result = Nd4j.create(getEmbeddingVectorSize());
    int i = 0;
    for(Encoder enc : encoders) {
      final INDArray vec = enc.encode(word);
      result.get(NDArrayIndex.interval(i, i + enc.getEmbeddingVectorSize())).assign(vec);
      i += enc.getEmbeddingVectorSize();
    }
    return result;
  }
  
  public INDArray encode(Iterable<? extends Span> spans) {
    INDArray result = Nd4j.create(getEmbeddingVectorSize());
    int i = 0;
    for(Encoder enc : encoders) {
      final INDArray vec = enc.encode(spans);
      result.get(NDArrayIndex.interval(i, i + enc.getEmbeddingVectorSize())).assign(vec);
      i += enc.getEmbeddingVectorSize();
    }
    return result;
  }
  
  @Override
  public INDArray encode(Span span) {
    return encode(span.toString());
  }
  
}
