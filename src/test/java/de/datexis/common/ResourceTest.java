package de.datexis.common;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class ResourceTest {
  
  @Test
  public void testInternalResource() {
    
    Resource res = Resource.fromJAR("encoder/word2vec.txt");
    assertTrue(res.exists());
//    assertTrue(res.isFile());
//    assertFalse(res.isDirectory());
    assertEquals("word2vec.txt", res.getFileName());
    
    res = Resource.fromJAR("encoder/a4et9gdkljsdg.txt");
    assertFalse(res.exists());
//    assertFalse(res.isFile()); // because it does not exist
//    assertFalse(res.isDirectory());
    assertEquals("a4et9gdkljsdg.txt", res.getFileName());
    
//    res = Resource.fromJAR("encoder");
//    assertTrue(res.exists());
//    assertFalse(res.isFile());
//    assertTrue(res.isDirectory());
//    assertEquals("", res.getFileName());
    
  }
  
}
