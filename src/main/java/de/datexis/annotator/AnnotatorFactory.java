package de.datexis.annotator;

import com.google.common.collect.Lists;
import de.datexis.common.ExternalResource;
import de.datexis.tagger.Tagger;
import de.datexis.common.Resource;
import de.datexis.encoder.Encoder;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Stream;
import javax.xml.bind.*;
import javax.xml.parsers.*;
import javax.xml.transform.*;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.reflections.Configuration;
import org.reflections.Reflections;
import org.reflections.scanners.TypeAnnotationsScanner;
import org.reflections.util.ConfigurationBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.*;
import org.w3c.dom.Element;
import org.xml.sax.SAXException;

/**
 * Factory Class for loading and saving Annotators
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class AnnotatorFactory {

  protected final static Logger log = LoggerFactory.getLogger(AnnotatorFactory.class);
  
  private final static String PACKAGE_REGEX = "^.+[\\.$]";
  
  // these lookup maps are used to identify class names without package information and even from old versions
  final static Map<String,Class<? extends Annotator>> annotatorClasses = new TreeMap<>();
  final static Map<String,Class<? extends AnnotatorComponent>> componentClasses = new TreeMap<>();
  
  static {
    Configuration conf = ConfigurationBuilder.build("de.datexis").setExpandSuperTypes(false);
    Reflections reflections = new Reflections(conf);
    annotatorClasses.put("Annotator", Annotator.class);
    Set<Class<? extends Annotator>> annotators = reflections.getSubTypesOf(Annotator.class);
    for(Class<? extends Annotator> c : annotators) annotatorClasses.put(c.getSimpleName(), c);         
    componentClasses.put("AnnotatorComponent", AnnotatorComponent.class);
    Set<Class<? extends AnnotatorComponent>> components = reflections.getSubTypesOf(AnnotatorComponent.class);
    for(Class<? extends AnnotatorComponent> c : components) componentClasses.put(c.getSimpleName(), c);         
  }
  
  /**
   * Writes an Annotator to a given Resource in XML format.
   * @param <A>
   * @param annotator
   * @param file The XML file
   */
  public static <A extends Annotator> void writeXML(A annotator, Resource file) {
    // We don't want to map the entire object as XML. Our file should 
    // only include the <annotator>s and <encoder>s with their respective 
    // models and configurations and serialize them as JSON.
    // XmlMapper mapper = new XmlMapper();
    // mapper.enable(SerializationFeature.INDENT_OUTPUT);
    // mapper.writeValueAsString(root)
    // So, instead we do it manually:
    try(OutputStream out = file.getOutputStream()) {
      
      DocumentBuilderFactory docFactory = DocumentBuilderFactory.newInstance();
      DocumentBuilder docBuilder = docFactory.newDocumentBuilder();
      Document xmlDoc = docBuilder.newDocument();
      
      Element xmlAnn = xmlDoc.createElement("annotator");
      xmlAnn.setAttribute("class", annotator.getClass().getSimpleName());
      xmlAnn.setAttribute("tagger", annotator.tagger.getId());
      JAXBContext provContext = JAXBContext.newInstance(Provenance.class);
      Provenance p = annotator.getProvenance();
      Marshaller m = provContext.createMarshaller();
      Element xmlProv = xmlDoc.createElement("provenance");
      m.marshal(p, xmlProv);
      xmlAnn.appendChild(xmlProv.getFirstChild());
      xmlDoc.appendChild(xmlAnn);
      
      writeComponent(annotator.tagger, xmlDoc, xmlAnn);
      for(AnnotatorComponent comp : annotator.components.values()) {
        writeComponent(comp, xmlDoc, xmlAnn);
      }
      
      TransformerFactory transformerFactory = TransformerFactory.newInstance();
      Transformer transformer = transformerFactory.newTransformer();
      transformer.setOutputProperty(OutputKeys.INDENT, "yes");
      transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "2");
      DOMSource source = new DOMSource(xmlDoc);
      StreamResult result =  new StreamResult(out);
      transformer.transform(source, result);
      out.flush();
      
    } catch (Exception ex) {
      log.error("Could not write XML: " + ex.toString());
    }
    
  }
  
  private static <C extends AnnotatorComponent> void writeComponent(C component, Document xmlDoc, Element xmlParent) {
    Element xmlComp = xmlDoc.createElement("component");
    xmlComp.setAttribute("class", component.getClass().getSimpleName());
    xmlComp.setAttribute("id", component.getId());
    xmlParent.appendChild(xmlComp);
    Element xmlModel = xmlDoc.createElement("model");
    xmlModel.setTextContent(component.getModel());
    xmlComp.appendChild(xmlModel);
    Element xmlConf = xmlDoc.createElement("conf");
    xmlConf.appendChild(xmlDoc.createCDATASection(component.getConf()));
    xmlComp.appendChild(xmlConf);
    if(component.getEncoders().iterator().hasNext()) {
      Element xmlEncoders = xmlDoc.createElement("encoders");
      for(Encoder e : component.getEncoders()) {
        Element xmlEncoder = xmlDoc.createElement("encoder");
        xmlEncoder.setAttribute("type", "input");
        xmlEncoder.setAttribute("id", e.getId());
        xmlEncoders.appendChild(xmlEncoder);
      }
      for(Encoder e : component.getTargetEncoders()) {
        Element xmlEncoder = xmlDoc.createElement("encoder");
        xmlEncoder.setAttribute("type", "target");
        xmlEncoder.setAttribute("id", e.getId());
        xmlEncoders.appendChild(xmlEncoder);
      }
      xmlComp.appendChild(xmlEncoders);
    }
  }

  @Deprecated //** please use loadAnnotator() */
  public static Annotator fromXML(Resource path) throws IOException {
    return fromXML(path, findXML(path));
  }

  public static Annotator loadAnnotator(Resource path, Resource... searchPaths) throws IOException {
    return fromXML(path, findXML(path), searchPaths);
  }
  
  @Deprecated //** please use loadAnnotator() */
  public static Annotator fromXML(Resource path, String name) throws IOException {
    return fromXML(path, name, new Resource[]{});
  }
  
  @Deprecated //** please use loadAnnotator() */
  public static Annotator fromXML(Resource path, String name, Resource... searchPaths) throws IOException {
    
    ObjectMapper mapper = new ObjectMapper();
    
    try(InputStream input = path.resolve(name).getInputStream()) {
      
      DocumentBuilderFactory docFactory = DocumentBuilderFactory.newInstance();
      DocumentBuilder docBuilder = docFactory.newDocumentBuilder();
      Document xmlDoc = docBuilder.parse(input);
      
      Element xmlAnn = xmlDoc.getDocumentElement();
      xmlAnn.normalize();
      if(!xmlAnn.getNodeName().equals("annotator")) throw new IllegalArgumentException("unknown file format");
      String expectedClass = xmlAnn.getAttribute("class").replaceFirst(PACKAGE_REGEX, "");
      if(!annotatorClasses.containsKey(expectedClass)) {
        throw new IllegalArgumentException("Annotator class " + expectedClass + " could not be found, please check model version and dependencies");
      }
      Annotator ann = annotatorClasses.get(expectedClass).newInstance();
      
      // create all components
      NodeList xmlComps = xmlAnn.getElementsByTagName("component");
      for(int i = 0; i < xmlComps.getLength(); i++) {
        // TODO: refactor next lines into function
        Element xmlComp = (Element) xmlComps.item(i); 
        String id = xmlComp.getAttribute("id");
        Element xmlModel = (Element) xmlComp.getElementsByTagName("model").item(0);
        String model = xmlModel.getTextContent();
        Element xmlConf = (Element) xmlComp.getElementsByTagName("conf").item(0);
        CDATASection json = (CDATASection) xmlConf.getFirstChild();
        expectedClass = xmlComp.getAttribute("class").replaceFirst(PACKAGE_REGEX, "");
        Class<?> clazz = componentClasses.get(expectedClass);
        // TODO: refactor next lines into function
        // TODO: teach this mapper to recognize "BIOESTag" without package. We did this before somewhere..
        AnnotatorComponent comp = (AnnotatorComponent) mapper.readValue(json.getData(), clazz);
        // try to load model from given paths
        if(model != null && !model.isEmpty()) {
          Resource modelPath = path.resolve(model);
          Resource absolutePath = new ExternalResource(model);
          boolean loaded = false;
          if(modelPath.exists()) {
            comp.loadModel(modelPath);
            loaded = true;
          } else if(absolutePath.exists()) {
            comp.loadModel(new ExternalResource(model));
            loaded = true;
          } else {
            for(Resource searchPath : searchPaths) {
              modelPath = searchPath.resolve(model);
              if(modelPath.exists()) {
                comp.loadModel(modelPath);
                loaded = true;
                break;
              }
            }
          }
          if(!loaded) /*!comp.isModelAvailable()*/
            throw new IllegalArgumentException("Model '" + model + "' not found for component " + comp.getName());
        }
        ann.components.put(id, comp);
      }
      
      // attach encoders to all components
      for(int i = 0; i < xmlComps.getLength(); i++) {
        Element xmlComp = (Element) xmlComps.item(i);
        AnnotatorComponent comp = ann.components.get(xmlComp.getAttribute("id"));
        NodeList xmlEncs = xmlComp.getElementsByTagName("encoders");
        if(xmlEncs.getLength() > 0) {
          xmlEncs = ((Element)xmlEncs.item(0)).getElementsByTagName("encoder");
          for(int j = 0; j < xmlEncs.getLength(); j++) {
            Element xmlEnc = (Element) xmlEncs.item(j); 
            Encoder enc = (Encoder) ann.components.get(xmlEnc.getAttribute("id"));
            if(xmlEnc.getAttribute("type").equals("input")) comp.addInputEncoder(enc);
            else if(xmlEnc.getAttribute("type").equals("target")) comp.addTargetEncoder(enc);
            else comp.addEncoder(enc);
          }
        }
      }
      
      // attach root tagger
      Tagger root = (Tagger) ann.components.get(xmlAnn.getAttribute("tagger"));
      ann.tagger = root;
      
      // attach provenance
      Element xmlProvenance = (Element) xmlAnn.getElementsByTagName("provenance").item(0);
      if(xmlProvenance != null) {
        JAXBContext provContext = JAXBContext.newInstance(Provenance.class);
        Unmarshaller m = provContext.createUnmarshaller();
        JAXBElement<Provenance> el = m.unmarshal(xmlProvenance, Provenance.class);
        if(!el.isNil()) ann.provenance = el.getValue();
      }
      
      return ann; 
      
    } catch (ParserConfigurationException | SAXException | JAXBException | InstantiationException | IllegalAccessException ex) {
      log.error("Could not read XML: " + ex.toString());
      ex.printStackTrace();
      return null;
    }
  }

  private static String findXML(Resource path) {
    try(Stream<Path> paths = Files.find(path.getPath(), 1, (file,attrs) -> attrs.isRegularFile() && file.toString().endsWith(".xml"))) {
      Optional<Path> p = paths.findFirst();
      if(p.isPresent()) return p.get().getFileName().toString();
      else return "annotator.xml";
    } catch (IOException ex) {
      log.warn("Could not find XML: " + ex.toString());
      return "annotator.xml";
    } catch (UnsupportedOperationException | NullPointerException ex) {
      // might be an internal resource, let's try "annotator.xml" for now
      return "annotator.xml";
    }
  }

  /** please use GenericMentionAnnotator.create() */
  @Deprecated 
  public static void createGenericMentionAnnotator() {
  }
  
}
