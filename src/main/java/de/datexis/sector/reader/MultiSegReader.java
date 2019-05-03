package de.datexis.sector.reader;

import de.datexis.common.InternalResource;
import de.datexis.common.Resource;
import de.datexis.model.*;
import de.datexis.preprocess.DocumentFactory;
import de.datexis.reader.RawTextDatasetReader;
import de.datexis.sector.model.SectionAnnotation;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.StandardCharsets;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.commons.io.LineIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Reader for MultiSeg dataset by Ivan Titov: "Multi-document topic segmentation". CIKM '10, Pages 1119-1128
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class MultiSegReader extends RawTextDatasetReader {

  protected final static Logger log = LoggerFactory.getLogger(MultiSegReader.class);
  
  @Override
  public Dataset read(Resource path) throws IOException {
    if(path.isDirectory()) {
      return readDatasetFromDirectory(path, "^(.+?)\\.(\\d+)$");
    } else if(path.isFile()) {
      Document doc = readDocumentFromFile(path);
      Dataset data = new Dataset(path.getFileName());
      data.addDocument(doc);
      return data;
    } else throw new FileNotFoundException("cannot open path: " + path.toString());
  }
  
  /**
   * Read a single Document from file.
   */
  @Override
  public Document readDocumentFromFile(Resource file) {

    try(InputStream in = file.getInputStream()) {
      CharsetDecoder utf8 = StandardCharsets.UTF_8.newDecoder();
      BufferedReader br = new BufferedReader(new InputStreamReader(in));
      Iterator<String> it = new LineIterator(br);
      int i = 0;
      int n = 0;
      
      Document doc = new Document();
      doc.setId(file.getFileName());
      doc.setSource(file.toString());
      doc.setType("multiseg");
      
      String line;
      StringBuilder text = new StringBuilder();
      
      Pattern filePattern = Pattern.compile("^(.+?/)(.+?)\\.(\\d+)$");
      Matcher m = filePattern.matcher(file.toString());
      if(!m.matches()) throw new IllegalArgumentException("invalid file name");
      
      String basePath =  m.group(1);
      String baseDoc = m.group(2);
      int docNum = Integer.parseInt(m.group(3));
      Resource labels = file instanceof InternalResource ? 
          Resource.fromJAR(basePath + baseDoc + ".label") : 
          Resource.fromFile(basePath, baseDoc + ".label");
      TreeSet[] sections = readSectionsFromLabel(labels, docNum);
      
      while(it.hasNext()) {
        
        line = it.next();
        
        if(sections[0].contains(i)) {
          // begin new section
          text = new StringBuilder();
        }  
        
        text.append(line).append("\n");
        
        if(sections[1].contains(i)) {
          // end the current section
          String sectionText = text.toString();
          if(sectionText.trim().length() > 0) {
            addToDocument(sectionText, n++, doc);
          }
          text = new StringBuilder();
        }
        
        i++;
        
      } 
      
      // end the last section - not required in case of begin-end
      /*String sectionText = text.toString();
      if(sectionText.trim().length() > 0) {
        addToDocument(sectionText, n++, doc);
      }*/

      return doc;
      
    } catch(IOException ex) {
      // IOException is now allowed in Stream
      log.error(ex.toString());
      throw new RuntimeException(ex.toString(), ex.getCause());
    }
    
  }

  private void addToDocument(String text, int sectionId, Document doc) {
    if(text.trim().length() == 0) return;
    //Document section = DocumentFactory.fromText(text, DocumentFactory.Newlines.KEEP);fromText(text, DocumentFactory.Newlines.KEEP);
    Document section = new Document();
    // we don't split sentences here but use entire paragraphs to stay comparable to the paper
    for(String paragraph : text.split("\n")) {
      //List<Token> tokens = DocumentFactory.createTokensFromText(paragraph.trim() + "\n");
      Document temp = DocumentFactory.fromText(paragraph.trim() + "\n", DocumentFactory.Newlines.KEEP);
      section.addSentence(DocumentFactory.createSentenceFromTokens(temp.getTokens()));
    }
    String sectionHead = Integer.toString(sectionId);
    doc.append(section);
    SectionAnnotation sectionAnn = new SectionAnnotation(Annotation.Source.GOLD, doc.getType(), sectionHead);
    sectionAnn.setSectionLabel(sectionHead); // heading is already set in constructor
    sectionAnn.setBegin(section.getBegin());
    sectionAnn.setEnd(section.getEnd());
    doc.addAnnotation(sectionAnn);
  }
  
  /**
   * Return a set of lines where a new sections starts.
   */
  protected TreeSet[] readSectionsFromLabel(Resource file, int docNum) throws IOException {
    TreeSet<Integer> sectionStarts = new TreeSet<>();
    TreeSet<Integer> sectionEnds = new TreeSet<>();
    //sectionStarts.add(0);
    List<Map.Entry<Integer, Integer>> sections = new ArrayList<>();
    try(InputStream in = file.getInputStream()) {
      CharsetDecoder utf8 = StandardCharsets.UTF_8.newDecoder();
      BufferedReader br = new BufferedReader(new InputStreamReader(in, utf8));
      String line = null;
      while ((line = br.readLine()) != null) {
        StringTokenizer tokens = new StringTokenizer(line, " ");
        String topicLabel = tokens.nextToken();
        while (tokens.hasMoreTokens()) {
          String token = tokens.nextToken();
          String[] segs = token.split("::", -1);
          int docId = Integer.parseInt(segs[0]);
          String[] segPoints = segs[1].split("-", -1);
          int start = Integer.parseInt(segPoints[0]);
          int end = Integer.parseInt(segPoints[1]);
          if(docId == docNum) {
            sectionStarts.add(start);
            sectionEnds.add(end);
            sections.add(new AbstractMap.SimpleEntry<>(start, end));
          }
        }
      }
    }
    return new TreeSet[] { sectionStarts, sectionEnds };
  }
  
  
}
