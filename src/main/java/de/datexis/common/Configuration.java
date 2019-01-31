package de.datexis.common;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Properties;
import org.reflections.util.ClasspathHelper;
import org.slf4j.LoggerFactory;

/**
 * Provides static mthods for configuration.
 * @author sarnold
 */
public class Configuration {

  private static final org.slf4j.Logger log = LoggerFactory.getLogger(Configuration.class);
  protected static final Properties config;
  
  static {
    log.info("Loading TeXoo configuration...");

    List<URL> propertyList = new ArrayList<>(ClasspathHelper.forResource("texoo.properties"));
    
    Properties fallback = new Properties();
    fallback.put("key", "default");
    config = new Properties(fallback);
    
    Collections.sort(propertyList, new Configuration.PropertyComparator());
    for(URL path : propertyList) {
      log.info("reading properties from {}", path.getFile());
      try(InputStream stream = new URL(path, "texoo.properties").openStream()) {
        config.load(stream);
      } catch(IOException ex) {
        log.warn("could not find properties file, continuing with defaults: " + ex.toString());
      }
    }
  }
  
  protected static class PropertyComparator implements Comparator<URL> {
    @Override
    public int compare(URL o1, URL o2) {
      if(o1.getPath().contains("texoo-core")) return -1;
      else if(o2.getPath().contains("texoo-core")) return 1;
      else return o1.getPath().compareTo(o2.getPath());
    }
  }
  
  /**
   * Read a property from texoo.properties
   * @param key
   * @return 
   */
  public static String getProperty(String key) {
    if(!config.containsKey(key)) log.warn("Configuration key '" + key + "' does not exist!");
    return config.getProperty(key, "");
  }
  
  public static String getVersion() {
    return config.getProperty("de.datexis.texoo.version", null);
  }
  
  public static boolean isCudaEnabled() {
    return config.getProperty("de.datexis.texoo.backend", "").contains("cuda");
  }
  
}
