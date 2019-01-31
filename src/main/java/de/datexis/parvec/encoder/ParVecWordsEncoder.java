package de.datexis.parvec.encoder;

import de.datexis.common.WordHelpers;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.model.Token;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Encodes a sentence as mean of individual Word Vectors.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class ParVecWordsEncoder extends ParVecEncoder {

  protected final static Logger log = LoggerFactory.getLogger(ParVecWordsEncoder.class);
  
  @Override
  public long getEmbeddingVectorSize() {
    // return size of the embedding!
    return layerSize;
  }
  
  @Override
  public INDArray encode(Span span) {
    if(span instanceof Sentence) {
      String text = ((Sentence)span).toTokenizedString()
              .trim()
              .replaceAll("\n", " *NL* ")
              .replaceAll("\t", " *t* ");
      INDArray sum = Nd4j.zeros(getEmbeddingVectorSize(), 1);
      int len = 0;
      for(String w : WordHelpers.splitSpaces(text)) {
        if(w.trim().isEmpty()) continue;
        INDArray arr = model.getWordVectorMatrix(preprocessor.preProcess(w));
        if(arr != null) {
          sum.addi(arr.transpose());
          len++;
        }
      }
      return len == 0 ? sum : sum.div(len);
    } else if(span instanceof Token) {
      INDArray arr = model.getWordVectorMatrix(preprocessor.preProcess(((Token)span).getText()));
      if(arr != null) return arr;
      else return Nd4j.zeros(getEmbeddingVectorSize(), 1);
    } else {
      return encode(span.getText());
    }
  }

}
