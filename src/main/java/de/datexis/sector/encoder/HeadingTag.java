package de.datexis.sector.encoder;

import de.datexis.encoder.Encoder;
import de.datexis.encoder.impl.BagOfWordsEncoder;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.tag.Tag;
import de.datexis.sector.model.SectionAnnotation;
import java.io.IOException;
import java.util.Iterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class HeadingTag implements Tag {

  protected final static Logger log = LoggerFactory.getLogger(HeadingTag.class);

  protected final String label;
  protected byte[] vector = null;
  protected final long length;
  protected double confidence = 0.;
  
  protected HeadingTag(String label, INDArray vector) {
    this.label = label;
    this.length = vector == null ? 0 : vector.length();
    try {
      if(vector != null) this.vector = Nd4j.toByteArray(vector);
    } catch(IOException ex) {
      log.error("IOError in HeadingTag: {}", ex.toString());
    }
  }
  
  @Override
  public String getTag() {
    return label;
  }

  @Override
  public String getTag(int index) {
    throw new UnsupportedOperationException("Not possible without encoder.");
  }
  
  @Override
  public int getVectorSize() {
    return (int) length;
  }

  @Override
  public INDArray getVector() {
    try {
      return Nd4j.fromByteArray(vector);
    } catch(IOException ex) {
      log.error("IOError in HeadingTag: {}", ex.toString());
      return null;
    }
  }

  @Override
  public double getConfidence() {
    return this.confidence;
  }
  
  public void setConfidence(double confidence) {
    this.confidence = confidence;
  }
  
  public static class Factory {
  
    protected final BagOfWordsEncoder encoder;
    
    public Factory(BagOfWordsEncoder encoder) {
      this.encoder = encoder;
    }
    
    public HeadingTag create(String heading) {
      return new HeadingTag(heading, encoder.encode(heading));
    }
    
    public HeadingTag create(INDArray prediction) {
      return new HeadingTag(encoder.getNearestNeighbour(prediction), prediction);
    }
    
    public void attachFromSectionAnnotations(Document doc, Annotation.Source source) {
      attachFromSectionAnnotations(doc, source, true);
    }
    
    public void attachFromSectionAnnotations(Document doc, Annotation.Source source, boolean attachVectors) {
      
      Iterator<SectionAnnotation> sections = doc.streamAnnotations(source, SectionAnnotation.class).sorted().iterator();
      
      if(doc.countSentences() <= 0) return;
      if(!sections.hasNext()) throw new IllegalArgumentException("Document has no sections");
      
      SectionAnnotation ann = sections.next();
      String title = ann.getSectionHeading();
      INDArray vec = attachVectors ? encoder.encode(title) : null;
      HeadingTag tag = new HeadingTag(title, vec);
      
      for(Sentence s : doc.getSentences()) {
        while(s.getBegin() >= ann.getEnd()) {
          if(sections.hasNext()) {
            ann = sections.next();
            title = ann.getSectionHeading();
            vec = attachVectors ? encoder.encode(title) : null;
            tag = new HeadingTag(title, vec);
          } else {
            log.error("Found Document with missing SectionAnnotations for Sentence position {}", s.getBegin());
          }
        }
        s.putTag(source, tag);
      }
      
      doc.setTagAvailable(source, HeadingTag.class, true);
      
    }
    
    public void attachFromSentenceVectors(Document doc, Class<? extends Encoder> encoder, Annotation.Source source) {
      for(Sentence s : doc.getSentences()) {
        s.putTag(source, create(s.getVector(encoder)));
      }
      doc.setTagAvailable(source, HeadingTag.class, true);
    }
    
  }

}
