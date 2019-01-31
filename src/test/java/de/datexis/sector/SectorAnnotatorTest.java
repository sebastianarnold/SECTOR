package de.datexis.sector;

import de.datexis.common.Resource;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.sector.model.SectionAnnotation;
import de.datexis.sector.reader.WikiSectionReader;
import java.io.IOException;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class SectorAnnotatorTest {
  
  public SectorAnnotatorTest() {
  }
  
  @Test
  public void testAnnotationEvaluation() throws IOException {
    Resource result = Resource.fromJAR("testdata/en_disease_higashi.json");
    Dataset data = WikiSectionReader.readDatasetFromJSON(result);
    assertEquals(1, data.countDocuments());
    Document doc = data.getDocument(0).get();

    System.out.println("GOLD");
    for(SectionAnnotation ann : doc.getAnnotations(Annotation.Source.GOLD, SectionAnnotation.class)) {
      System.out.println(ann.getBegin() + "\t" + ann.getSectionLabel());
    }

    System.out.println("PRED");
    for(SectionAnnotation ann : doc.getAnnotations(Annotation.Source.PRED, SectionAnnotation.class)) {
      System.out.println(ann.getBegin() + "\t" +  ann.getSectionLabel() + "\t" + ann.getConfidence());
    }
    
  }
  
}
