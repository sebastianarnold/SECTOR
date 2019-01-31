package de.datexis.common;


import de.datexis.common.Configuration;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class PropertiesTest {
  
  @Test
  public void testProperties() {
    String version = Configuration.getVersion();
    assertNotNull("Could not find de.datexis.texoo.version in Properties. Please update and check your texoo.properties!", version);
    assertFalse(version.equals("${project.version}"));
    System.out.println("TeXoo Version: " + version);
  }
 
  
}
