package de.datexis.sector.reader;

import de.datexis.common.Resource;
import de.datexis.model.*;
import de.datexis.preprocess.DocumentFactory;
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
 * Reader for Wiki-727l Dataset by Omri Koshorek, Adir Cohen, Noam Mor, Michael
 * Rotman, and Jonathan Berant. 2018. "Text segmentation as a supervised learning task."
 * In Proceedings of the 2018 Conference of the North American Chapter of the Association for
 * Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), volume 2,
 * pages 469–473.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class Wiki727Reader extends RawTextDatasetReader {

  protected final static Logger log = LoggerFactory.getLogger(Wiki727Reader.class);
  
  protected int sectionLevel = 2;
  protected boolean skipPrefaceText = false;
  protected boolean skipPrefaceAnnotation = false;
  
  protected Pattern SECTION_PATTERN = Pattern.compile("^========,(\\d+),(.+?)\\.$");
  
  /**
   * Create Annotations down to a given level.
   * 0 - include all section
   * 1 - the whole document <h1> as one section
   * 2 - include <h2> sections
   * 3 - include <h3> subsections
   * etc.
   */
  public Wiki727Reader withSectionLevel(int level) {
    this.sectionLevel = level;
    return this;
  }
  
  public Wiki727Reader withSkipPreface(boolean skip) {
    this.skipPrefaceText = skip;
    return this;
  }
  
  /**
   * Read a single Document from file.
   */
  @Override
  public Document readDocumentFromFile(Resource file) {

    try(InputStream in = file.getInputStream()) {
      CharsetDecoder utf8 = StandardCharsets.UTF_8.newDecoder();
      BufferedReader br = new BufferedReader(new InputStreamReader(in, utf8));
      Iterator<String> it = new LineIterator(br);
      int k = 0;
      int j = 0;
      int length = 0;
      
      Document doc = new Document();
      doc.setId(file.getFileName());
      doc.setSource(file.toString());
      doc.setType("wiki");
      
      String line;
      StringBuilder text = new StringBuilder();
      SectionAnnotation ann = new SectionAnnotation(Annotation.Source.GOLD);
      String sectionHeading = "";
      
      while(it.hasNext()) {
        
        line = it.next();
        Matcher matcher = SECTION_PATTERN.matcher(line);
        
        if(line.startsWith("=====") && matcher.matches()) {
          int level = Integer.parseInt(matcher.group(1));
          String heading = matcher.group(2);
          
          // check heading depth
          if(sectionLevel == 0 || level <= sectionLevel) {
            
            // end the current section
            String sectionText = text.toString();
            if(sectionText.trim().length() > 0) {
              addToDocument(sectionText, sectionHeading, doc);
            }
            
            // rebuild heading for next section
            int split = 0;
            while(--level > 1) split = sectionHeading.indexOf(" | ", split + 1);
            if(split > 0) sectionHeading = sectionHeading.substring(0, split) + " | "; // remove
            else if(split < 0) sectionHeading = sectionHeading + " | "; // add
            else sectionHeading = ""; // reset
            sectionHeading += heading;
            
            text = new StringBuilder();
            
          }
          
        } else {
          
          if(text.length() > 0) text.append(" ");
          line = line
              .replaceAll("\\*\\*\\*LIST\\*\\*\\*", "")
              .replaceAll("\\*\\*\\*formula\\*\\*\\*", "")
              .replaceAll("\\*\\*\\*codice\\*\\*\\*", "");
          if(!line.trim().isEmpty()) {
            text.append(line).append("\n");
          }
          
        }
        
      } 
      
      // end the last section
      String sectionText = text.toString();
      if(sectionText.trim().length() > 0) {
        addToDocument(sectionText, sectionHeading, doc);
      }

      return doc;
      
    } catch(IOException ex) {
      // IOException is now allowed in Stream
      log.error(ex.toString());
      throw new RuntimeException(ex.toString(), ex.getCause());
    }
    
  }

  
  private void addToDocument(String text, String sectionHeading, Document doc) {
    if(text.trim().length() == 0) return;
    //Document section = DocumentFactory.fromText(text, DocumentFactory.Newlines.DISCARD);
    Document section = new Document();
    // we don't split sentences here but use entire paragraphs to stay comparable to the paper
    for(String paragraph : text.split("\n")) {
      //List<Token> tokens = DocumentFactory.createTokensFromText(paragraph.trim() + "\n");
      if(paragraph.trim().isEmpty()) continue;
      Document temp = DocumentFactory.fromText(paragraph.trim() + "\n", DocumentFactory.Newlines.KEEP);
      section.addSentence(DocumentFactory.createSentenceFromTokens(temp.getTokens()));
    }
    if(sectionHeading == null) return;
    String sectionHead = sectionHeading.replaceFirst("\\|.+$","").trim().toLowerCase();
    if(skipPrefaceText && sectionHead.equals("preface")) {
      //doc.setAbstract(section.getText());
    } else {
      doc.append(section);
      SectionAnnotation sectionAnn = new SectionAnnotation(Annotation.Source.GOLD, doc.getType(), sectionHeading);
      sectionAnn.setSectionLabel(sectionHeading); // heading is already set in constructor
      sectionAnn.setBegin(section.getBegin());
      sectionAnn.setEnd(section.getEnd());
      doc.addAnnotation(sectionAnn);
    }
  }
  
  

}
