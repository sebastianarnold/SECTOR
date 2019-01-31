package de.datexis.common;


import de.datexis.common.WordHelpers;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Token;
import de.datexis.preprocess.DocumentFactory;
import de.datexis.preprocess.MinimalLowercasePreprocessor;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.junit.Test;
import static org.junit.Assert.*;
import org.junit.Before;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class WordHelpersTest {
  
  private final static String[][] de = {
    { "Auch die Ehrungen muß man anmelden FRANKFURT A. M. Langvermählte Paare und die ältesten Bürger haben in Frankfurt ein Anrecht auf Ehrungen bei Jubiläumsfeiern.",
      "auch die ehrungen muss man anmelden frankfurt a m langvermaehlte paare und die aeltesten buerger haben in frankfurt ein anrecht auf ehrungen bei jubilaeumsfeiern" },
    { "Leitung: Manfred Lochner, Tel. 0 61 81 / 1 30 93 26.",
      "leitung manfred lochner tel # # # # # # #" },
    { "Das Ergebnis sollte voriges Jahr als Broschüre zum 725. Stadt-Geburtstag erscheinen.",
      "das ergebnis sollte voriges jahr als broschuere zum # stadt-geburtstag erscheinen" },
    { "Um mit Belze Berta (1906 – 1991) zu reden: \" Mir saan net gefräit worn: Was willste lerne?",
      "um mit belze berta # - # zu reden mir saan net gefraeit worn was willste lerne" },
    { "1. Februar 2003 (01.02.2003 unter Zugrundelegung der im deutschsprachigen Raum weiterhin gebräuchlichen Reihenfolge nach DIN 1355-1)" ,
      "# februar # # unter zugrundelegung der im deutschsprachigen raum weiterhin gebraeuchlichen reihenfolge nach din #-#" },
    { "Seit 2007 gibt es den Kyron mit dem robusten XDi270-Dieselmotor, was ihm eine Leistung von 121 kW und ein Drehmoment von 340 Nm verleiht.",
      "seit # gibt es den kyron mit dem robusten xdi#-dieselmotor was ihm eine leistung von # kw und ein drehmoment von # nm verleiht" },
    { "Zhang Dejiang (chinesisch 張德江 / 张德江, Pinyin Zhāng Déjiāng ; * November 1946 in Tai'an, Provinz Liaoning, Republik China) ist ein chinesischer kommunistischer Politiker." ,
      "zhang dejiang chinesisch pinyin zhang dejiang november # in taian provinz liaoning republik china ist ein chinesischer kommunistischer politiker" },
    { "Der „Löwe des Nordostens“ (Leão do Nordeste), wie der Verein auch gerne bezeichnet wird, spielte nach dem Abstieg seit 2013 wieder in der Série B, der zweithöchsten Spielklasse Brasiliens.",
      "der loewe des nordostens leao do nordeste wie der verein auch gerne bezeichnet wird spielte nach dem abstieg seit # wieder in der serie b der zweithoechsten spielklasse brasiliens" },
    { "Die DIN 5008 schreibt für Zeitangaben den Doppelpunkt vor: 12:30 Uhr. Der Duden erlaubt zusätzlich den Punkt und die Hochstellung: 12.30 Uhr oder 1230 Uhr.",
      "die din # schreibt fuer zeitangaben den doppelpunkt vor # uhr der duden erlaubt zusaetzlich den punkt und die hochstellung # uhr oder # uhr" }
    };

  @Before
  public void init() {
  }
  
  @Test
  public void testStopWords() {
    for(WordHelpers.Language lang : WordHelpers.Language.values()) {
      WordHelpers utils = new WordHelpers(lang);
      assertFalse(utils.getStopWords().isEmpty());
    }
    TokenPreProcess pre = new MinimalLowercasePreprocessor();
    WordHelpers utils = new WordHelpers(WordHelpers.Language.DE);
    Document doc = DocumentFactory.fromText(de[0][0]);
    Document doc2 = new Document();
    for(Sentence s : doc.getSentences()) {
      Sentence s2 = new Sentence();
      for(Token t : s.getTokens()) {
        if(!utils.isStopWord(t.getText(), pre)) {
          Token t2 = new Token(t.getText());
          s2.addToken(t2);
        }
      }
      doc2.addSentence(s2, true);
    }
    assertEquals("Ehrungen man anmelden FRANKFURT A. M. Langvermählte Paare ältesten Bürger haben Frankfurt Anrecht Ehrungen Jubiläumsfeiern", doc2.getText());
  }
  
  @Test
  public void testMinimalLowercasePreprocessor() {
    TokenPreProcess pre = new MinimalLowercasePreprocessor();
    for(String[] test : de) {
      Document doc = DocumentFactory.fromText(test[0]);
      assertEquals(test[0], doc.getText());
      for(Sentence s : doc.getSentences()) {
        for(Token t : s.getTokens()) {
          t.setText(pre.preProcess(t.getText()));
          t.setBegin(0);
          t.setLength(t.getText().length());
        }
      }
      assertEquals(test[1], doc.getText());
    }
  }
  
  @Test
  public void wordOverlapTest() {
    //                                        0   1   2    3    4    5         6
    Sentence s1 = DocumentFactory.fromText("This text is about word overlapping.").getSentence(0);
    Sentence s2 = DocumentFactory.fromText("This text is  not about word overlapping.").getSentence(0);
    assertEquals(36, WordHelpers.getSpanOverlapLength(s1, s2));
    assertEquals( 4, WordHelpers.getSpanOverlapLength(s1.getToken(0), s2.getToken(0)));
    assertEquals( 4, WordHelpers.getSpanOverlapLength(s2.getToken(0), s1.getToken(0)));
    assertEquals( 4, WordHelpers.getSpanOverlapLength(s1.getToken(1), s2.getToken(1)));
    assertEquals( 3, WordHelpers.getSpanOverlapLength(s1.getToken(3), s2.getToken(3)));
    assertEquals( 0, WordHelpers.getSpanOverlapLength(s1.getToken(1), s2.getToken(2)));
    assertEquals( 4, WordHelpers.getSpanOverlapLength(s1.getToken(4), s2.getToken(4)));
    assertEquals( 4, WordHelpers.getSpanOverlapLength(s1.getToken(5), s2.getToken(5)));
    assertEquals( 6, WordHelpers.getSpanOverlapLength(s1.getToken(5), s2.getToken(6)));
    assertEquals( 1, WordHelpers.getSpanOverlapLength(s1.getToken(6), s2.getToken(6)));
    assertEquals( 0, WordHelpers.getSpanOverlapLength(s1.getToken(0), s2.getToken(6)));
    assertEquals( 0, WordHelpers.getSpanOverlapLength(s1.getToken(6), s2.getToken(0)));
  }
  
}
