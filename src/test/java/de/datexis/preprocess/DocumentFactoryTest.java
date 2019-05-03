package de.datexis.preprocess;

import de.datexis.model.Document;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class DocumentFactoryTest {

  protected final static Logger log = LoggerFactory.getLogger(DocumentFactoryTest.class);

  protected final String text = "Dementia\nSeveral specific diagnostic criteria can be used to diagnose vascular dementia, including the Diagnostic and Statistical Manual of "
          + "Mental Disorders, Fourth Edition (DSM-IV) criteria, the International Classification of Diseases, Tenth Edition (ICD-10) criteria, the National "
          + "Institute of Neurological Disorders and Stroke criteria, Association Internationale pour la Recherche et l'Enseignement en Neurosciences (NINDS-AIREN) "
          + "criteria, the Alzheimer's Disease Diagnostic and Treatment Center criteria, and the Hachinski Ischemic Score (after Vladimir Hachinski).\nThe recommended "
          + "investigations for cognitive impairment include: blood tests (for anemia, vitamin deficiency, thyrotoxicosis, infection, etc.), chest X-Ray, ECG, "
          + "and neuroimaging, preferably a scan with a functional or metabolic sensitivity beyond a simple CT or CTT. When available as a diagnostic tool, single "
          + "photon emission computed tomography (SPECT) and positron emission tomography (PET) neuroimaging may be used to confirm a diagnosis of multi-infarct "
          + "dementia in conjunction with evaluations involving mental status examination tests. In a person already having dementia, SPECT appears to be superior in "
          + "differentiating multi-infarct dementia from Alzheimer's disease, compared to the usual mental testing and medical history analysis.";
  
  protected final String expected = "Dementia Several specific diagnostic criteria can be used to diagnose vascular dementia, including the Diagnostic and Statistical Manual of "
          + "Mental Disorders, Fourth Edition (DSM-IV) criteria, the International Classification of Diseases, Tenth Edition (ICD-10) criteria, the National "
          + "Institute of Neurological Disorders and Stroke criteria, Association Internationale pour la Recherche et l'Enseignement en Neurosciences (NINDS-AIREN) "
          + "criteria, the Alzheimer's Disease Diagnostic and Treatment Center criteria, and the Hachinski Ischemic Score (after Vladimir Hachinski). The recommended "
          + "investigations for cognitive impairment include: blood tests (for anemia, vitamin deficiency, thyrotoxicosis, infection, etc.), chest X-Ray, ECG, "
          + "and neuroimaging, preferably a scan with a functional or metabolic sensitivity beyond a simple CT or CTT. When available as a diagnostic tool, single "
          + "photon emission computed tomography (SPECT) and positron emission tomography (PET) neuroimaging may be used to confirm a diagnosis of multi-infarct "
          + "dementia in conjunction with evaluations involving mental status examination tests. In a person already having dementia, SPECT appears to be superior in "
          + "differentiating multi-infarct dementia from Alzheimer's disease, compared to the usual mental testing and medical history analysis.";
  
  @Test
  public void testSentenceSplitting() {
    Document doc;
    doc = DocumentFactory.fromText(text, DocumentFactory.Newlines.DISCARD);
    Assert.assertEquals(5, doc.countSentences());
    Assert.assertEquals("Dementia", doc.getSentence(0).getText());
    Assert.assertEquals("Several", doc.getSentence(1).getToken(0).getText());
    Assert.assertEquals("The", doc.getSentence(2).getToken(0).getText());
    Assert.assertEquals("When", doc.getSentence(3).getToken(0).getText());
    Assert.assertEquals("In", doc.getSentence(4).getToken(0).getText());
    Assert.assertEquals(expected, doc.getText());
  }
  
  @Test
  public void testNewLines() {
    Document doc;
    doc = DocumentFactory.fromText(text, DocumentFactory.Newlines.KEEP);
    Assert.assertEquals(5, doc.countSentences());
    Assert.assertEquals(text, doc.getText());
    Assert.assertEquals("Dementia\n", doc.getSentence(0).getText());
  }
  
  @Test
  public void testSentenceBoundaries() {
    String text = "Human rights in Tanzania.\nThe issue of human rights in Tanzania, a nation with a 2012 population of 44,928,923, is hard. In its 2013 Freedom in the World report, Freedom House declared the country \"Partly Free\".\nHuman rights concerns.\nThe United Nations Human Rights Council in October 2011 at its meeting in Geneva completed a Universal Periodic Review (UPR) of the human rights situation in Tanzania. At this UPR, the United Nations Country Team (UNCT) and several countries addressed various problems in Tanzania.\nGender equality.\nNational reviews and assessments of equality between men and women... have identified a range of challenges..., which continue to prevail. These include the persistent and increasing burden of poverty on women; inequalities in arrangements for productive activities and in access to resources; inequalities in the sharing of power and decision-making; lack of respect for and inadequate promotion and protection of the human rights of women; and inequalities in managing natural resources and safeguarding the environment.... Particular attention should be drawn to the widespread marginalization of the girl child in different spheres of life, including education, and the total exclusion caused for many by early and forced marriage.... Gender-based violence is prevalent.";
    Document doc = DocumentFactory.fromText(text,DocumentFactory.Newlines.DISCARD);
    //Assert.assertEquals(11, doc.countSentences());
    Assert.assertEquals(9, doc.countSentences()); // "...." are not detected as Sentence boundaries
    Assert.assertEquals(text.replace("\n", " "), doc.getText());
  }
  
  @Test
  public void testEscapedChars() {
    String text = "Anah.\nAnah or Ana (, \"\u02be\u0100na\"), formerly also known as Anna, is an Iraqi town on the Euphrates river, approximately midway between the Gulf of Alexandretta and the Persian Gulf. Anah lies from west to east on the right bank along a bend of the river just before it turns south towards Hit.\nName.\nThe town is called Ha-na-at in a Babylonian letter around 2200\u00a0, A-na-at by the scribes of Tukulti-Ninurta \u00a0, and An-at by the scribes of Assur-nasir-pal II in 879\u00a0. The name has been connected with the widely worshipped war goddess Anat. It was known as \"Anath\u014d\" () to Isidore Charax and ' to Ammianus Marcellinus; early Arabic writers described it variously as \"\u02be\u0100na\" or (as if plural) \"\u02be\u0100n\u0101t\".\nAncient.\nDespite maintaining its name across 42 centuries, the exact location of the settlement seems to have moved from time to time. Sources across most of its early history, however, place Anah on an island in the Euphrates.\nIts early history under the Babylonians is uncertain. A 3rd-millennium\u00a0 letter mentions six \"men of Hanat\" are mentioned in a description of disturbances in the Residency of Suhi, which would have included the district of Anah. It is probably not the place mentioned by Amenhotep I in the 16th century\u00a0 or in the speech of Sennacherib's messengers to Hezekiah, but probably was the site \"in the middle of the Euphrates\" opposite which Assur-nasir-pal II halted during his 879\u00a0 campaign.";
    Document doc = DocumentFactory.fromText(text,DocumentFactory.Newlines.DISCARD);
    Assert.assertEquals(text.replace("\n", " ").replace("\u00a0", " "), doc.getText());
  }
  
  @Test
  public void testDoubleNewlines() {
    String text = "sentence.\n\nEEG:\nEEG Lorem ipsum dolor";
    Document doc = DocumentFactory.fromText(text,DocumentFactory.Newlines.KEEP);
    Assert.assertEquals(text, doc.getText());
    Assert.assertEquals("sentence.\n\n", doc.getSentence(0).getText());
  }
  
}
