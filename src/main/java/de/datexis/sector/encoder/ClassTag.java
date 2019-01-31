package de.datexis.sector.encoder;

import com.fasterxml.jackson.annotation.JsonIgnore;
import de.datexis.encoder.Encoder;
import de.datexis.encoder.LookupCacheEncoder;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.tag.Tag;
import de.datexis.sector.model.SectionAnnotation;
import java.io.IOException;
import java.util.Objects;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A Tag that is used to label a single Class.
 * @author sarnold
 */
public class ClassTag implements Tag {

  private static final Logger log = LoggerFactory.getLogger(ClassTag.class);

  protected final String label;
  protected final int numClasses;
  protected final double[] vector;
  protected double confidence = 0.;
  protected final int index;
  
  public ClassTag(String label, INDArray vector) {
    this.vector = vector.transpose().toDoubleVector();
    this.label = label;
    this.index = getMaxIndex(vector);
    this.confidence = vector.maxNumber().doubleValue();
    this.numClasses = (int) vector.length();
  }
  
  /*public ClassTag(String label) {
    this(label, Nd4j.create(1));
  }
  
  public ClassTag() {
    this("", Nd4j.create(1));
  }*/
 
  @JsonIgnore
  private int getMaxIndex(INDArray v) {
    double max = Double.MIN_VALUE;
    int index = 0;
    double d;
    for(int j=0; j<v.length(); j++) {
      d = v.getDouble(j);
      if(d > max) {
        index = j;
        max = d;
      }
    }
    return index;
	}
  
  public int getIndex() {
    return index;
  }
  
  @Override
  public double getConfidence() {
    return this.confidence;
  }
  
  public ClassTag setConfidence(double confidence) {
    this.confidence = confidence;
    return this;
  }
  
  @Override
  public int getVectorSize() {
    return numClasses;
  }
  
  /**
   * @return The predicted Vector for this Ta
   */
  @Override
  public INDArray getVector() {
    return Nd4j.create(vector).transposei();
  }
  
  @Override
  public String toString() {
    return label;
  }
  
  @Override
  public String getTag() {
    return label;
  }
  
  @Override
  public String getTag(int index) {
    throw new UnsupportedOperationException("not implemented yet");
  }

  public boolean matches(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }
    final ClassTag other = (ClassTag) obj;
    if (!Objects.equals(this.label, other.label)) {
      return false;
    }
    return true;
  }

  @Override
  public boolean equals(Object o) {
    if(this == o) {
      return true;
    }
    if(!(o instanceof ClassTag)) {
      return false;
    }
    ClassTag classTag = (ClassTag) o;
    return Objects.equals(numClasses, classTag.numClasses) &&
           Objects.equals(label, classTag.label);
  }

  @Override
  public int hashCode() {
    return Objects.hash(label, numClasses);
  }
  
  public static class Factory {
  
    protected final LookupCacheEncoder encoder;
    
    public Factory(LookupCacheEncoder encoder) {
      this.encoder = encoder;
    }
    
    public ClassTag create(String heading) {
      return new ClassTag(heading, encoder.oneHot(heading));
    }
    
    public ClassTag create(INDArray prediction) {
      return new ClassTag(encoder.getNearestNeighbour(prediction), prediction);
    }
    
    public void attachFromSectionAnnotations(Document doc, Annotation.Source source) {
      for(SectionAnnotation ann : doc.getAnnotations(source, SectionAnnotation.class)) {
        String clss = ann.getSectionLabel();
        INDArray vec = encoder.oneHot(clss);
        doc.streamSentencesInRange(ann.getBegin(), ann.getEnd(), false).forEach(
          s -> s.putTag(source, new ClassTag(clss, vec))
        );
      }
      doc.setTagAvailable(source, ClassTag.class, true);
    }
    
    public void attachFromSentenceVectors(Document doc, Class<? extends Encoder> encoder, Annotation.Source source) {
      for(Sentence s : doc.getSentences()) {
        s.putTag(source, create(s.getVector(encoder)));
      }
      doc.setTagAvailable(source, ClassTag.class, true);
    }
    
  }
  
}
