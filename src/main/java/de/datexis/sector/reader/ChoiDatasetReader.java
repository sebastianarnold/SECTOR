package de.datexis.sector.reader;

import de.datexis.common.Resource;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.preprocess.DocumentFactory;
import de.datexis.reader.RawTextDatasetReader;
import de.datexis.sector.model.SectionAnnotation;
import org.apache.commons.io.LineIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Reader for Text Segmentation datasets by Freddy Choi (2000)
 * "Advances in domain independent linear text segmentation". In Proceedings of NAACL'00.
 * https://github.com/logological/C99
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class ChoiDatasetReader extends RawTextDatasetReader {

  protected final static Logger log = LoggerFactory.getLogger(ChoiDatasetReader.class);
  
  protected final static String SEGMENT_SPLIT = "==========";
  
  /**
   * Read a single Document from file.
   */
  @Override
  public Document readDocumentFromFile(Resource file) {

    try(InputStream in = file.getInputStream()) {
      CharsetDecoder utf8 = StandardCharsets.UTF_8.newDecoder();
      BufferedReader br = new BufferedReader(new InputStreamReader(in, utf8));
      Iterator<String> it = new LineIterator(br);
      int j = 0;
      
      Document doc = new Document();
      doc.setId(file.getFileName());
      doc.setSource(file.toString());
      doc.setType("seg");
      
      String line;
      StringBuilder text = new StringBuilder();
      SectionAnnotation ann = new SectionAnnotation(Annotation.Source.GOLD);
      String sectionHeading = "";
      
      while(it.hasNext()) {
        
        line = it.next();
        
        if(line.equals(SEGMENT_SPLIT)) {
            
          // end the current section
          String sectionText = text.toString();
          if(sectionText.trim().length() > 0) {
            addToDocument(sectionText, doc);
          }
          text = new StringBuilder();
          
        } else {
          
          if(text.length() > 0) text.append(" ");
          if(!line.trim().isEmpty()) {
            if(!(line.endsWith(".") || line.endsWith("!") || line.endsWith("?"))) line = line + ".";
            text.append(line).append("\n");
          }
          
        }
        
      } 
      
      // end the last section
      String sectionText = text.toString();
      if(sectionText.trim().length() > 0) {
        addToDocument(sectionText, doc);
      }

      return doc;
      
    } catch(IOException ex) {
      // IOException is now allowed in Stream
      log.error(ex.toString());
      throw new RuntimeException(ex.toString(), ex.getCause());
    }
    
  }

  
  private void addToDocument(String text, Document doc) {
    if(text.trim().length() == 0) return;
    //Document section = DocumentFactory.fromText(text, DocumentFactory.Newlines.DISCARD);
    Document section = new Document();
    // we don't split sentences here but use entire paragraphs to stay comparable to the dataset
    for(String paragraph : text.split("\n")) {
      if(paragraph.trim().isEmpty()) continue;
      Document temp = DocumentFactory.fromText(paragraph.trim() + "\n", DocumentFactory.Newlines.KEEP);
      section.addSentence(DocumentFactory.createSentenceFromTokens(temp.getTokens()));
    }
    doc.append(section);
    String sectionHeading = Integer.toString(section.getBegin()); // create artificial heading
    SectionAnnotation sectionAnn = new SectionAnnotation(Annotation.Source.GOLD, doc.getType(), sectionHeading);
    sectionAnn.setSectionLabel(sectionHeading); // heading is already set in constructor
    sectionAnn.setBegin(section.getBegin());
    sectionAnn.setEnd(section.getEnd());
    doc.addAnnotation(sectionAnn);
  }
  
  /**
   * Read .ref/.pred files produced by C99.
   */
  public static void readC99Result(Dataset dataset, Resource path) throws IOException {
    
    Iterator<Path> files = Files.walk(path.getPath())
          .filter(p -> Files.isRegularFile(p, LinkOption.NOFOLLOW_LINKS))
          .filter(p -> p.getFileName().toString().endsWith(".pred"))
          .sorted()
          .iterator();
    
    Pattern pattern = Pattern.compile("\\/(\\d+)(.ref)?.pred$");
    
    while(files.hasNext()) {
      String f = files.next().toString();
      Matcher matcher = pattern.matcher(f);
      matcher.find();
      int docId = Integer.parseInt(matcher.group(1));
      log.info("reading doc id {} from file {}", docId, f);
      //if(Integer.parseInt(f.replace(".ref.pred", "")) != docId) log.warn("invalid doc id");
      Resource file = Resource.fromFile(f);
      try(InputStream in = file.getInputStream()) {
        CharsetDecoder utf8 = StandardCharsets.UTF_8.newDecoder();
        BufferedReader br = new BufferedReader(new InputStreamReader(in, utf8));
        Iterator<String> it = new LineIterator(br);
        int k = 0;
        int j = 0;
        int length = 0;
        Document doc = dataset.getDocument(docId).get();
        SectionAnnotation ann = new SectionAnnotation(Annotation.Source.PRED);
        while(it.hasNext()) {
          String line = it.next();
          if(line.equals(SEGMENT_SPLIT)) {
            j++;
            if(length > 0) {
              doc.addAnnotation(ann);
            }
            ann = new SectionAnnotation(Annotation.Source.PRED);
            ann.setSectionLabel(Integer.toString(j));
            length = 0;
          } else {
            Sentence s = doc.getSentence(k);
            if(length == 0) {
              ann.setBegin(s.getBegin());
            } else {
              ann.setEnd(s.getEnd());
            }
            length++;
            k++;
            if(!s.getText().trim().equals(line.trim())) {
              log.warn("docId {} k={} different sentences\n{}\n{}", docId, k, line, s.getText());
            }
          }
        }
        //docId++;
        //dataset.addDocument(doc);
      }
    }
    
  }
  

}
