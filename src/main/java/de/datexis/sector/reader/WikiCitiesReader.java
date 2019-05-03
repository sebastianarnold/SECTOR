package de.datexis.sector.reader;

import de.datexis.common.Resource;
import de.datexis.model.*;
import de.datexis.preprocess.DocumentFactory;
import de.datexis.reader.DatasetReader;
import de.datexis.reader.RawTextDatasetReader;
import de.datexis.sector.model.SectionAnnotation;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.commons.io.LineIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Reader for WikiCities dataset by Harr Chen, S. R. K. Branavan, Regina Barzilay,
 * and David R. Karger. 2009. "Global models of document structure using latent permutations."
 * In Proceedings of Human Language Technologies: The 2009 Annual Conference of the North
 * American Chapter of the Association for Computational Linguistics, pages 371–379. ACL.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class WikiCitiesReader implements DatasetReader {

  protected final static Logger log = LoggerFactory.getLogger(WikiCitiesReader.class);
  
  protected boolean skipTopLevelSegment = true;
  
  protected Pattern LINE_PATTERN = Pattern.compile("^(\\d+),(\\d+),(.+?)(.+?)$");
  protected String TOPLEVEL_STRING = "TOP-LEVEL SEGMENT";
  
  public WikiCitiesReader withSkipTopLevelSegment(boolean skip) {
    this.skipTopLevelSegment = skip;
    return this;
  }
  
  /**
   * Read a single Document from file.
   */
  @Override
  public Dataset read(Resource file) throws IOException {

    try(InputStream in = file.getInputStream()) {
      CharsetDecoder utf8 = StandardCharsets.UTF_8.newDecoder();
      BufferedReader br = new BufferedReader(new InputStreamReader(in, utf8));
      Iterator<String> it = new LineIterator(br);
      
      Dataset result = new Dataset(file.getFileName());
      result.setName(file.getFileName());
      Document doc = new Document();
      
      String line;
      StringBuilder text = new StringBuilder();
      String sectionHeading = "";
      
      while(it.hasNext()) {
        
        line = it.next();
        Matcher matcher = LINE_PATTERN.matcher(line);
        
        if(!matcher.matches()) {
          log.error("matcher did not match for lineL\n{}", line);
          continue;
        }
        
        String documentNo = matcher.group(1);
        int sentenceNo = Integer.parseInt(matcher.group(2));
        String heading = matcher.group(3);
        String sentence = matcher.group(4);
          
        if(sentenceNo == 1) {
          
          // end the last section
          String sectionText = text.toString();
          if(sectionText.trim().length() > 0) {
            addToDocument(sectionText, sectionHeading, doc);
          }
          
          // end the last document
          if(doc.countTokens() > 0) {
            result.addDocument(doc);
          }
          
          //start new document
          doc = new Document();
          doc.setId(documentNo);
          text = new StringBuilder();
          sectionHeading = "";
          
        }
        
        if(skipTopLevelSegment && heading.equals(TOPLEVEL_STRING)) continue;
          
        if(!heading.equals(sectionHeading)) {
          
          // end the last section
          String sectionText = text.toString();
          if(sectionText.trim().length() > 0) {
            addToDocument(sectionText, sectionHeading, doc);
          }
          
          // start new section
          text = new StringBuilder();
          sectionHeading = heading;
          
        }
        
        if(text.length() > 0) text.append(" ");
        text.append(sentence).append(" .");
        
      } 
      
      // end the last section
      String sectionText = text.toString();
      if(sectionText.trim().length() > 0) {
        addToDocument(sectionText, sectionHeading, doc);
      }

      // end the last document
      if(doc.countTokens() > 0) {
        result.addDocument(doc);
      }
          
      return result;
      
    } catch(IOException ex) {
      // IOException is now allowed in Stream
      log.error(ex.toString());
      throw new RuntimeException(ex.toString(), ex.getCause());
    }
    
  }

  
  private void addToDocument(String text, String sectionHeading, Document doc) {
    
    if(text.trim().length() == 0) return;
    Document section = DocumentFactory.fromTokenizedText(text);
    if(sectionHeading == null) sectionHeading = "";
    else if(sectionHeading.equals(TOPLEVEL_STRING)) sectionHeading = "preface";
    else sectionHeading = sectionHeading.trim().toLowerCase();
    
    doc.append(section);
    SectionAnnotation sectionAnn = new SectionAnnotation(Annotation.Source.GOLD, "wiki", sectionHeading);
    sectionAnn.setSectionLabel(sectionHeading.replaceAll("\\s+", "_"));
    sectionAnn.setBegin(section.getBegin());
    sectionAnn.setEnd(section.getEnd());
    doc.addAnnotation(sectionAnn);

  }

}
