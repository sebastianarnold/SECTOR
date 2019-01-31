package de.datexis.encoder;

import de.datexis.model.Span;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * An Encoder converts text (Span) to embedding vectors (INDArray).
 * E.g. word embedding
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public interface IEncoder {
  
  /**
	 * Get the size of the embedding vector
	 * @return INDArray vector length
	 */
	public long getEmbeddingVectorSize();
  
  /**
	 * Generate a fixed-size vector of a String
	 * @param word
	 * @return Mx1 column vector (INDArray) containing the encoded String
	 */
	public abstract INDArray encode(String word);
  
  /**
	 * Generate a fixed-size vector of a single Span
   * @param span the Span to encode
	 * @return Mx1 column vector (INDArray) containing the encoded Span
	 */
	public abstract INDArray encode(Span span);
  
  /**
   * Encode a fixed-size vector from multiple Spans
   * @param spans the Spans to encode
   * @return Mx1 column vector (INDArray) containing all Spans combined (e.g. average)
   */
  public INDArray encode(Iterable<? extends Span> spans);
  
}
