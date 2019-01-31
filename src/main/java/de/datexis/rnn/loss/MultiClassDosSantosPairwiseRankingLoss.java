package de.datexis.rnn.loss;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Extended for "Multi Class Multi Label Problems" Implementation of ranking loss function of (dos Santos et al., 2015)
 * https://arxiv.org/abs/1504.06580
 * L = log(1 + e^(gamma(mPlus - Score(x)yPlus))) + log(1 + e^(gamma(mMinus + Score(x)cMinus)))
 */
public class MultiClassDosSantosPairwiseRankingLoss implements ILossFunction {

  protected static final Logger log = LoggerFactory.getLogger(MultiClassDosSantosPairwiseRankingLoss.class);

  private Number positiveClassExclusionFactor = -1000000;
  private Number gamma = 2;
  private Number mPlus = 2.5;
  private Number mMinus = 0.5;
  
  public MultiClassDosSantosPairwiseRankingLoss() {
  }

  public MultiClassDosSantosPairwiseRankingLoss(Number gamma, Number mPlus, Number mMinus) {
    this.gamma = gamma;
    this.mPlus = mPlus;
    this.mMinus = mMinus;
  }

  public MultiClassDosSantosPairwiseRankingLoss(double gamma, double mPlus, double mMinus, double positiveClassExclusionFactor) {
    this.gamma = gamma;
    this.mPlus = mPlus;
    this.mMinus = mMinus;
    this.positiveClassExclusionFactor = positiveClassExclusionFactor;
  }


  
  /*
  L = log(1 + e^(gamma(mPlus - avg(Score(x)yPlus)))) + log(1 + e^(gamma(mMinus + Score(x)cMinus)))
 */

  public INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
    INDArray scoreArr;
    INDArray output = activationFn.getActivation(preOutput.dup(), true);

    INDArray mMinuss = Nd4j.zeros(output.shape()).addi(mMinus);

    INDArray positiveExamples = output.mul(labels);
    positiveExamples = averageOverCorrectClasses(labels, positiveExamples);
    INDArray positiveExp = positiveExamples.rsubi(mPlus).muli(gamma);
    INDArray positiveE = Transforms.exp(positiveExp);
    INDArray leftLog = Transforms.log(positiveE.addi(1));

    INDArray negativeExp = mMinuss.addi(output).muli(gamma);
    INDArray negativeE = Transforms.exp(negativeExp);
    INDArray negativeExamples = output.addi(labels.mul(positiveClassExclusionFactor));
    INDArray maxNeg = negativeExamples.argMax(1);
    INDArray rightLog = Transforms.log(negativeE.addi(1));

    INDArray negWithHighestScore = Nd4j.zeros(preOutput.shape());
    for(int i = 0; i < maxNeg.length(); i++) {
      int index = maxNeg.getInt(i);
      negWithHighestScore.put(i, index, 1);
    }
    
    leftLog = leftLog.muli(Transforms.min(labels.sum(1),1));
    rightLog = rightLog.muli(negWithHighestScore);

    scoreArr = leftLog.addi(rightLog.sum(1));

    if(mask != null) {
      scoreArr.muliColumnVector(mask);
    }

    return scoreArr;
  }

  private INDArray averageOverCorrectClasses(INDArray labels, INDArray positiveExamples) {
    INDArray correctClassesByExample = Transforms.max(labels.sum(1), 1); // fix for examples without label
    return positiveExamples.sum(1).divi(correctClassesByExample);
  }

  @Override
  public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
    INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);

    double score = scoreArr.sumNumber().doubleValue();

    if(average) {
      score /= scoreArr.size(0);
    }

    return score;
  }

  @Override
  public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
    INDArray scoreArray = scoreArray(labels, preOutput, activationFn, mask);

    return scoreArray.sum(1);
  }

  /* 
  L = log(1 + e^(gamma(mPlus -x))) + log(1 + e^(gamma(mMinus + x )))
  d/dx = -(e^(gamma (mPlus - x)) gamma)/(1 + e^(gamma (mPlus - x))) +
   (e^(gamma (mMinus + x)) gamma)/(1 + e^(gamma (mMinus + x)))
   
   -(e^(2 (2.5 - x)) 2)/(1 + e^(2 (2.5 -  x))) 
    +    (e^(2 (0.5 + y)) 2)/(1 + e^(2 (0.5 + y))) 
   */
  @Override
  public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
    INDArray output = activationFn.getActivation(preOutput.dup(), true);

    INDArray dlDx = computeDlDx(labels, output);

    //Everything below remains the same
    output = activationFn.backprop(preOutput.dup(), dlDx).getFirst();
    //multiply with masks, always
    if(mask != null) {
      output.muliColumnVector(mask);
    }

    return output;
  }

  public INDArray computeDlDx(INDArray labels, INDArray predictedScores) {
    INDArray mPluss = Nd4j.zeros(predictedScores.shape()).addi(mPlus);
    INDArray mMinuss = Nd4j.zeros(predictedScores.shape()).addi(mMinus);

    INDArray gammaDivNumCorrectLabels = Transforms.max(labels.sum(1), 1).rdivi(gamma);
    INDArray positiveExamples = predictedScores.mul(labels);

    INDArray leftHandExp = Transforms.exp((positiveExamples.sum(1).rsubi(mPlus)).muli(gammaDivNumCorrectLabels));
    INDArray leftHandNumerator = leftHandExp.mulColumnVector(gammaDivNumCorrectLabels);
    INDArray leftHandDenominator = leftHandExp.addi(1);
    INDArray leftHand = leftHandNumerator.divi(leftHandDenominator);
    leftHand = labels.mulColumnVector(leftHand.negi());

    INDArray rightHandExp = Transforms.exp((mMinuss.addi(predictedScores)).muli(gamma));
    INDArray rightHandNumerator = rightHandExp.mul(gamma);
    INDArray rightHandDenominator = rightHandExp.addi(1);
    INDArray rightHand = rightHandNumerator.divi(rightHandDenominator);

    predictedScores = predictedScores.addi(labels.mul(positiveClassExclusionFactor));

    INDArray maxNegative = predictedScores.argMax(1);
    INDArray negWithHighestScoreMask = Nd4j.zeros(predictedScores.shape());
    for(int i = 0; i < maxNegative.length(); i++) {
      int index = maxNegative.getInt(i);
      negWithHighestScoreMask.put(i, index, 1);
    }

    leftHand = leftHand.muli(labels);
    rightHand = rightHand.muli(negWithHighestScoreMask);
    rightHand = rightHand.mulColumnVector(Transforms.min(labels.sum(1),1)); // ignore examples without correct label.


    return leftHand.addi(rightHand);
  }

  @Override
  public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
    return new Pair<>(
      computeScore(labels, preOutput, activationFn, mask, average),
      computeGradient(labels, preOutput, activationFn, mask));
  }

  @Override
  public String name() {
    return this.getClass().getSimpleName();
  }

  public Number getGamma() {
    return gamma;
  }

  public void setGamma(Number gamma) {
    this.gamma = gamma;
  }

  public Number getmPlus() {
    return mPlus;
  }

  public void setmPlus(Number mPlus) {
    this.mPlus = mPlus;
  }

  public Number getmMinus() {
    return mMinus;
  }

  public void setmMinus(Number mMinus) {
    this.mMinus = mMinus;
  }

  public Number getPositiveClassExclusionFactor() {
    return positiveClassExclusionFactor;
  }

  public void setPositiveClassExclusionFactor(Number positiveClassExclusionFactor) {
    this.positiveClassExclusionFactor = positiveClassExclusionFactor;
  }
}
