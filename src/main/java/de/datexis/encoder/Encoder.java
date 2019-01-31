package de.datexis.encoder;

import com.google.common.collect.Lists;

import de.datexis.annotator.AnnotatorComponent;
import org.nd4j.linalg.api.ndarray.INDArray;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.model.Token;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.nd4j.linalg.factory.Nd4j;

/**
 * An Encoder converts text (Span) to embedding vectors (INDArray).
 * E.g. word embedding
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public abstract class Encoder extends AnnotatorComponent implements IEncoder {

  public Encoder() {
    this("");
  }

  public Encoder(String id) {
    super(false);
    this.id = id;
  }
    
  /**
   * Encode a fixed-size vector from multiple Spans
   * @param spans the Spans to encode
   * @return INDArray containing all Tokens combined
   */
  public INDArray encode(Iterable<? extends Span> spans) {
    INDArray avg = Nd4j.create(getEmbeddingVectorSize(), 1);
    INDArray vec;
    int i = 0;
    for(Span s : spans) {
      vec = encode(s.getText());
      if(vec != null) {
        avg.addi(vec);
        i++;
      }
    }
    return avg.divi(i);
  }

  /**
   * Encodes each element in the input and attaches the vectors to the element.
   * Please override this if the elements of your encoders are not independent or stateful.
   * @param input - the Document that should be encoded
   * @param elementClass - the class of sub elements in the Document, e.g. Sentence.class
   */
  public void encodeEach(Document input, Class<? extends Span> elementClass) {
    if(elementClass == Token.class) input.streamTokens().forEach(t -> t.putVector(this.getClass(), encode(t)));
    else if(elementClass == Sentence.class) input.streamSentences().forEach(s -> s.putVector(this.getClass(), encode(s)));
    else throw new IllegalArgumentException("Cannot encode class " + elementClass.toString() + " from Document");
  }

  /**
   * Encodes each element in the input and returns these vectors as matrix.
   * Please override this if the elements of your encoders are not independent or stateful.
   *  @param input - the Document that should be encoded
   * @param timeStepClass - the class of sub elements in the Document, e.g. Sentence.class
   */
  public INDArray encodeMatrix(List<Document> input, int maxTimeSteps, Class<? extends Span> timeStepClass) {

    INDArray encoding = Nd4j.zeros(input.size(), getEmbeddingVectorSize(), maxTimeSteps);
    Document example;

    for(int batchIndex = 0; batchIndex < input.size(); batchIndex++) {
      
      example = input.get(batchIndex);

      List<? extends Span> spansToEncode = Collections.EMPTY_LIST;
      if(timeStepClass == Token.class) spansToEncode = Lists.newArrayList(example.getTokens());
      else if(timeStepClass == Sentence.class) spansToEncode = Lists.newArrayList(example.getSentences());

      for(int t = 0; t < spansToEncode.size() && t < maxTimeSteps; t++) {
        INDArray vec = encode(spansToEncode.get(t));
        //encoding.put(new INDArrayIndex[] {point(batchIndex), all(), point(t)}, vec);
        encoding.getRow(batchIndex).getColumn(t).assign(vec); // this one is faster
      }
      
    }
    return encoding;
  }

  /**
   * Encodes each element in the input and attaches the vectors to the element.
   * Please override this if the elements of your encoders are not independent or stateful.
   * Please override this if your encoder allows batches.
   * @param docs - the Documents that should be encoded
   * @param elementClass - the class of sub elements in the Document, e.g. Sentence.class
   */
  public void encodeEach(Collection<Document> docs, Class<? extends Span> elementClass) {
    for(Document doc : docs) {
      encodeEach(doc, elementClass);
    }
  }
  
  /**
   * Encodes each element in the input and attaches the vectors to the element.
   * Please override this if the elements of your encoders are not independent or stateful.
   * @param input - the Sentence that should be encoded
   * @param elementClass - the class of sub elements in the Sentence, e.g. Token.class
   */
  public void encodeEach(Sentence input, Class<? extends Span> elementClass) {
    if(elementClass == Token.class) input.streamTokens().forEach(t -> t.putVector(this.getClass(), encode(t)));
    else throw new IllegalArgumentException("Cannot encode class " + elementClass.toString() + " from Sentence");
  }
    
  public abstract void trainModel(Collection<Document> documents);
  
  public void trainModel(Stream<Document> documents) {
    trainModel(documents.collect(Collectors.toList()));
  }
  
}
