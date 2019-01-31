package de.datexis.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import de.datexis.encoder.Encoder;
import de.datexis.encoder.EncoderSet;
import de.datexis.model.tag.Tag;
import java.io.IOException;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.LoggerFactory;

/**
 * Span of Characters in the Document.
 * @author sarnold
 */
// Disabled because Tokens and Sentences are identified by their holders, e.g. "sentences:[...]"
// If this needs to be enabled - please check why exactly! DocumentModelTest will fail
//@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "class")
//@JsonSubTypes({@JsonSubTypes.Type(value = Token.class), @JsonSubTypes.Type(value = Sentence.class)})
public abstract class Span implements Comparable<Span> {
  
  protected static final org.slf4j.Logger log = LoggerFactory.getLogger(Span.class);
  
  /**
   * Reference to the Document that this Span belongs to.
   */
  private Document documentRef;
  
  /**
   * The cursor positions of the Span in the Document (exclusive end)
   */
	protected int begin, end;
  
  /**
   * The unique ID of this span (e.g. database primary key)
   */
  protected Long uid = null;
  
  /**
   * Encoded column vectors of this Span. Only initialized when used.
   */
  private Map<String,byte[]> vectors = null;
  
  /**
   * List of Tags that were assigned to this Span from Gold, Prediction or User sources.
   * Only initialized when used.
   */
  private EnumMap<Annotation.Source, Map<String,Object>> tags = null;
  
  public Span() {}
  
  /**
   * @return reference to the Document that this Span belongs to
   */
  @JsonIgnore
  public Document getDocumentRef() {
    return documentRef;
  }
  
  //@JsonInclude(JsonInclude.Include.NON_NULL)
  @JsonIgnore
  @Deprecated
  public Long getDocumentRefUid() {
    return this.getDocumentRef().getUid();
  }
  
  public void setDocumentRef(Document doc) {
    this.documentRef = doc;
  }
  
  /**
   * @return the cursor position before the beginning of the Span
   */
  public int getBegin() {
    return begin;
  }

  public void setBegin(int begin) {
    this.begin = begin;
  }

   /**
   * @return the cursor position after the end of the Span
   */
  @JsonIgnore
  public int getEnd() {
    return end;
  }
  
  public void setEnd(int end) {
    this.end = end;
  }
  
 /**
   * @return the length of the Span
   */
  public int getLength() {
    return getEnd() - getBegin();
  }
  
  public void setLength(int length) {
    this.end = this.begin + length;
  }
  
  public void setUid(Long uid) {
    this.uid = uid;
  }
  
  @JsonInclude(JsonInclude.Include.NON_NULL)
  public Long getUid() {
    return this.uid;
  }
  
  /**
   * @return the Text of this span
   */
  public abstract String getText();
  
  /**
   * Add an INDArray from an Encoder to this Span.
   * Existing vectors of the same class will be overridden.
   * @param type The Encoder class that generated the vector.
   * @param vec  The Vector itself. Will be cached in memory.
   */
  public void putVector(Class<? extends Encoder> type, INDArray vec) {
    putVector(type.getCanonicalName(), vec);
  }
  
  /**
   * Add an INDArray to this Span.
   * Existing vectors with same identifier will be overridden.
   * @param identifier An identifier for this vector.
   * @param vec  The column vector itself. Will be duplicated to heap.
   */
  public void putVector(String identifier, INDArray vec) {
    if(vectors == null) vectors = new TreeMap<>();
    try {
      vectors.put(identifier, Nd4j.toByteArray(vec));
    } catch(IOException ex) {
      log.error("IOError in putVector(): {}", ex.toString());
    }
  }
  
  /**
   * Clear all Vectors that are cached in this Span.
   */
  public void clearVectors() {
    if(vectors == null) return;
    for(String key : vectors.keySet().toArray(new String[0])) {
      vectors.remove(key);
    }
  }
  
  /**
   * Clear all Vectors of a given type cached in this Span.
   */
  public void clearVectors(Class<? extends Encoder> type) {
    clearVectors(type.getCanonicalName());
  }
  
  /**
   * Clear all Vectors of a given identifier cached in this Span.
   */
  public void clearVectors(String identifier) {
    if(vectors == null) return;
    vectors.remove(identifier);
  }
  
  /**
   * Get the Vector/Embedding added to this Span. If no Vector was added, return null.
   * @param type The Encoder class that generated the vector.
   * @return A previously added INDArray or null
   */
  public INDArray getVector(Class<? extends Encoder> type) {
    return getVector(type.getCanonicalName());
  }
  
  /**
   * Get the Vector/Embedding added to this Span. If no Vector was added, return null.
   * @param identifier The identifier for this vector.
   * @return A previously added INDArray or null
   */
  public INDArray getVector(String identifier) {
    if(vectors != null && vectors.containsKey(identifier)) {
      try {
        final byte[] vec = vectors.get(identifier);
        return Nd4j.fromByteArray(vec);
      } catch(IOException ex) {
        log.error("IOError in putVector(): {}", ex.toString());
        return null;
      }
    } else {
      log.error("Requesting unknown vector with identifier '" + identifier + "'");
      return null;
    }
  }
  
  public boolean hasVector(Class<? extends Encoder> type) {
    return hasVector(type.getCanonicalName());
  }
  
  public boolean hasVector(String identifier) {
    return vectors != null && vectors.containsKey(identifier);
  }
  
  /**
   * Concatenate all vectors to create a feature vector.
   * @param encoders The Encoders to use
   * @return A feature vector which is a concatenation of all Encoders
   */  
  public INDArray getVector(EncoderSet encoders) {
    //TODO: better use Nd4j.hstack(arrs)
    INDArray result = Nd4j.create(encoders.getEmbeddingVectorSize());
    int i = 0;
    for(Encoder enc : encoders) {
      INDArray vec = getVector(enc.getClass());
      result.get(NDArrayIndex.interval(i, i + enc.getEmbeddingVectorSize())).assign(vec);
      i += enc.getEmbeddingVectorSize();
    }
    return result;
  }
    
  public <T extends Tag> Span putTag(Annotation.Source source, T tag) {
    if(tags == null) tags = new EnumMap<>(Annotation.Source.class);
    if(!tags.containsKey(source)) tags.put(source, new HashMap<>(4, 1.0f));
    tags.get(source).put(tag.getClass().getCanonicalName(), tag);
    return this;
  }
  
  /**
   * Clear all Tags that are assigned to this Span.
   */
  public void clearTags(Annotation.Source source) {
    if(tags == null) return;
    if(!tags.containsKey(source)) return;
    tags.get(source).clear();
    tags.remove(source);
  }
  
  
  /**
   * Returns a Tag for this Span. If no tag exists, a standard tag (e.g. BIO2Tag.O()) is returned.
   * @param <T>
   * @param source
   * @param type
   * @return 
   */
  public <T extends Tag> T getTag(Annotation.Source source, Class<T> type) {
    try {
      if(tags != null && tags.get(source) != null) {//return (T) tags.get(source).getOrDefault(type, type.newInstance());
        Map<String, Object> map = tags.get(source);
        T result = (T) map.get(type.getCanonicalName());
        if(result != null) return result;
        else return type.newInstance();
      } else return type.newInstance();
    } catch(InstantiationException | IllegalAccessException ex) {
      return null;
    }
  }

  /**
   * Span ordering based on begin and end positions.
   */
  @Override
  public int compareTo(Span other) {
    int c = (this.begin - other.begin);
    if(c == 0) c = (this.end - other.end);
    return c;
  }
  
  @Override
  public boolean equals(Object o) {
    if(this == o) {
      return true;
    }
    if(!(o instanceof Span)) {
      return false;
    }
    Span span = (Span) o;
    return Objects.equals(getBegin(),span.getBegin()) &&
           Objects.equals(getEnd(), span.getEnd()) &&
           Objects.equals(tags, span.tags);
  }

  @Override
  public int hashCode() {
    return Objects.hash(getBegin(), getEnd(), tags);
  }
}
