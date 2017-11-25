#****************************************************************
#
# evaluate.py - the evaluation program.
#
# Author: Yue Zhang
#
# Computing lab, University of Oxford. 2006.11
#
#****************************************************************

#================================================================
#
# Import modules.
#
#================================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "libs")))
import getopt

#----------------------------------------------------------------
#
# addTuples - add two tuples element by element.
#
# Inputs:  tuple1 - operand1
#          tuple2 - operand2
#
# Returns: tuple
#
#----------------------------------------------------------------

def addTuples(tuple1, tuple2):
   return tuple([tuple1[i]+tuple2[i] for i in xrange(len(tuple1))])

#----------------------------------------------------------------
#
# addListToList - add the second list to the first list.
#
# Inputs:  list1 - operand1, the one modified list
#          list2 - operand2, the list added
#
#----------------------------------------------------------------

def addListToList(list1, list2):
   for i in xrange(len(list1)):
      list1[i] += list2[i]

#----------------------------------------------------------------
#
# subtractListFromList - subtract the second list from the first list.
#
# Inputs:  list1 - operand1, the one modified list
#          list2 - operand2, the list added
#
#----------------------------------------------------------------

def subtractListFromList(list1, list2):
   for i in xrange(len(list1)):
      list1[i] -= list2[i]

#----------------------------------------------------------------
#
# dotProduct - compute the dot-product for lists/tuples.
#
# Inputs:  list1 - operand1
#          list2 - operand2
#
# Returns: int
#
#----------------------------------------------------------------

def dotProduct(list1, list2):
   nReturn = 0
   for i in xrange(len(list1)):
      nReturn += list1[i] * list2[i]
   return nReturn

#----------------------------------------------------------------
#
# addDictToDict - add a dictionary with int value to another dictionary
#
# Input: dict1 - operand1, the one that is added to
#        dict2 - operand2, the one to add
#
# Example:
# addDictToDict({'a':1,'b':2}, {'b':1,'c':2}) = {'a':1,'b':4,'c':2}
#
#----------------------------------------------------------------

def addDictToDict(dict1, dict2):
   for key in dict2:
      if key in dict1:
         dict1[key] += dict2[key]
      else:
         dict1[key] = dict2[key]

#----------------------------------------------------------------
#
# subtractDictFromDict - subtract a dictionary with int value from another dictionary
#
# Input: dict1 - operand1, the one that is modified to
#        dict2 - operand2, the one to substract
#
# Example:
# subtractDictFromDict({'a':1,'b':2}, {'b':1,'c':2}) = {'a':1,'b':1,'c':-2}
#
#----------------------------------------------------------------

def subtractDictFromDict(dict1, dict2):
   for key in dict2:
      if key in dict1:
         dict1[key] -= dict2[key]
      else:
         dict1[key] = -dict2[key]

#================================================================
#
# CRawSentenceReader - the raw sentence reader
#
# This reader is aimed for Chinese.
#
#================================================================

class CRawSentenceReader(object):

   #----------------------------------------------------------------
   #
   # __init__ - initialisation
   #
   # Inputs: sPath - the file for reading
   #
   #----------------------------------------------------------------

   def __init__(self, sPath, sEncoding="utf-8"):
      self.m_sPath = sPath
      self.m_oFile = open(sPath)
      self.m_sEncoding = sEncoding

   #----------------------------------------------------------------
   #
   # __del__ - destruction
   #
   #----------------------------------------------------------------

   def __del__(self):
      self.m_oFile.close()

   #----------------------------------------------------------------
   #
   # readNonEmptySentence - read the next sentence
   #
   # Returns: list of characters or None if the EOF symbol met.
   #
   #----------------------------------------------------------------

   def readNonEmptySentence(self):
      # 1. read one line
      sLine = "\n"                              # use a pseudo \n to start
      while sLine:                              # while there is a line
         sLine = sLine.strip()                  # strip the line
         if sLine:                              # if the line isn't empty
            break                               # break
         sLine = self.m_oFile.readline()        # read next line
         if not sLine:                          # if eof symbol met
            return None                         # return
      # 2. analyse this line
      uLine = sLine.decode(self.m_sEncoding)    # find unicode
      lLine = [sCharacter.encode(self.m_sEncoding) for sCharacter in uLine]
      return lLine

   #----------------------------------------------------------------
   #
   # readSentence - read the next sentence
   #
   # Returns: list of characters or None if the EOF symbol met.
   #
   #----------------------------------------------------------------

   def readSentence(self):
      # 1. read one line
      sLine = self.m_oFile.readline()           # read next line
      if not sLine:                             # if eof symbol met
         return None                            # return
      # 2. analyse this line
      uLine = sLine.strip().decode(self.m_sEncoding)    # find unicode
      lLine = [sCharacter.encode(self.m_sEncoding) for sCharacter in uLine]
      return lLine

#================================================================
#
# CPennTaggedSentenceReader - the tagged sentence reader
#
#================================================================

class CPennTaggedSentenceReader(object):

   #----------------------------------------------------------------
   #
   # __init__ - initialisation
   #
   # Inputs: sPath - the file for reading
   #
   #----------------------------------------------------------------

   def __init__(self, sPath):
      self.m_sPath = sPath
      self.m_oFile = open(sPath)

   #----------------------------------------------------------------
   #
   # __del__ - destruction
   #
   #----------------------------------------------------------------

   def __del__(self):
      self.m_oFile.close()

   #----------------------------------------------------------------
   #
   # readNonEmptySentence - read the next sentence
   #
   # Input: bIgnoreNoneTag - ignore _-NONE- tagged word?
   #
   # Returns: list of word, tag pairs or None if the EOF symbol met.
   #
   #----------------------------------------------------------------

   def readNonEmptySentence(self, bIgnoreNoneTag):
      # 1. read one line
      sLine = "\n"                              # use a pseudo \n to start
      while sLine:                              # while there is a line
         sLine = sLine.strip()                  # strip the line
         if sLine:                              # if the line isn't empty
            break                               # break
         sLine = self.m_oFile.readline()        # read next line
         if not sLine:                          # if eof symbol met
            return None                         # return
      # 2. analyse this line
      lLine = sLine.strip().split(" ")
      lNewLine = []
      for nIndex in xrange(len(lLine)):
         tTagged = tuple(lLine[nIndex].split("_"))
         assert(len(tTagged)<3)
         if len(tTagged)==1:
            tTagged = (tTagged[0], "-NONE-")
         if (bIgnoreNoneTag==False) or (tTagged[0]): # if we take -NONE- tag, or if we find that the tag is not -NONE-
            lNewLine.append(tTagged)
      return lNewLine


   #----------------------------------------------------------------
   #
   # readNonEmptySentence - read the next sentence
   #
   # Input: bIgnoreNoneTag - ignore _-NONE- tagged word?
   #
   # Returns: list of word, tag pairs or None if the EOF symbol met.
   #
   #----------------------------------------------------------------

   def readSentence(self, bIgnoreNoneTag):
      # 1. read one line
      sLine = self.m_oFile.readline()           # read next line
      if not sLine:                             # if eof symbol met
         return None                            # return
      # 2. analyse this line
      lLine = sLine.strip().split(" ")
      lNewLine = []
      for nIndex in xrange(len(lLine)):
         tTagged = tuple(lLine[nIndex].split("_"))
         assert(len(tTagged)<3)
         if len(tTagged)==1:
            tTagged = (tTagged[0], "-NONE-")
         if (bIgnoreNoneTag==False) or (tTagged[0]): # if we take -NONE- tag, or if we find that the tag is not -NONE-
            lNewLine.append(tTagged)
      return lNewLine

#================================================================
#
# Global.
#
#================================================================

g_sInformation = "\nThe evaluation program for Chinese Tagger. \n\n\
  Yue Zhang 2006\n\
  Computing laboratory, Oxford\n\n\
evaluate.py candidate_text reference_text\n\n\
The candidate and reference text need to be files with tagged sentences. Each sentence takes one line, and each word is in the format of Word_Tag.\n\n\
"

#----------------------------------------------------------------
#
# evaluateSentence - evaluate one sentence
#
# Input: tCandidate - candidate sentence
#        tReference
#
# Return: int for correct words
#
#----------------------------------------------------------------

def evaluateSentence(lCandidate, lReference):
   nCorrectWords = 0
   nCorrectTags = 0
   nChar = 0
   indexCandidate = 0
   indexReference = 0
   while lCandidate and lReference:
      if lCandidate[0][0] == lReference[0][0]:  # words right
         nCorrectWords += 1
         if lCandidate[0][1] == lReference[0][1]: # tags
            nCorrectTags += 1
         indexCandidate += len(lCandidate[0][0]) # move
         indexReference += len(lReference[0][0])
         lCandidate.pop(0)
         lReference.pop(0)
      else:
         if indexCandidate == indexReference:
            indexCandidate += len(lCandidate[0][0]) # move
            indexReference += len(lReference[0][0])
            lCandidate.pop(0)
            lReference.pop(0)
         elif indexCandidate < indexReference:
            indexCandidate += len(lCandidate[0][0])
            lCandidate.pop(0)
         elif indexCandidate > indexReference:
            indexReference += len(lReference[0][0]) # move
            lReference.pop(0)
   raw_l = max(indexCandidate, indexReference)
   total_num = (raw_l + 1) * raw_l / 2
   return nCorrectWords, nCorrectTags, total_num


def evaluateSentence_boundaries(lCandidate, lReference):
   nCorrectWords = 0
   nCorrectTags = 0
   indexCandidate = 0
   indexReference = 0
   while len(lCandidate) > 1 and len(lReference) > 1:
      if lCandidate[0][0] == lReference[0][0]:  # words right
         nCorrectWords += 1
         if lCandidate[0][1] == lReference[0][1] and lCandidate[1][1] == lReference[1][1]: # tags
            nCorrectTags += 1
         indexCandidate += len(lCandidate[0][0]) # move
         indexReference += len(lReference[0][0])
         lCandidate.pop(0)
         lReference.pop(0)
      else:
         if indexCandidate == indexReference:
            nCorrectTags += 1
            if lCandidate[0][1] == lReference[0][1] and lCandidate[1][1] == lReference[1][1]:  # tags
                nCorrectTags += 1
            indexCandidate += len(lCandidate[0][0]) # move
            indexReference += len(lReference[0][0])
            lCandidate.pop(0)
            lReference.pop(0)
         elif indexCandidate < indexReference:
            indexCandidate += len(lCandidate[0][0])
            lCandidate.pop(0)
         elif indexCandidate > indexReference:
            indexReference += len(lReference[0][0]) # move
            lReference.pop(0)
   return nCorrectWords, nCorrectTags

#================================================================
#
# score.
#
#================================================================

def readNonEmptySentenceList(sents, bIgnoreNoneTag=True):
    out = []
    for sent in sents:
        lNewLine = []
        lLine = sent.split(' ')
        for nIndex in range(len(lLine)):
            tTagged = tuple(lLine[nIndex].split("_"))
            assert (len(tTagged) < 3)
            if len(tTagged) == 1:
                tTagged = (tTagged[0], "-NONE-")
            if (bIgnoreNoneTag == False) or (tTagged[0]):  # if we take -NONE- tag, or if we find that the tag is not -NONE-
                lNewLine.append(tTagged)
        out.append(lNewLine)
    return out


def score(sReference, sCandidate, tag_num=1, verbose=False):
    nTotalCorrectWords = 0
    nTotalCorrectTags = 0
    nTotalPrediction = 0
    nCandidateWords = 0
    nReferenceWords = 0
    reference = readNonEmptySentenceList(sReference)
    candidate = readNonEmptySentenceList(sCandidate)
    assert len(reference) == len(candidate)
    for lReference, lCandidate in zip(reference, candidate):
        n = len(lCandidate)
        nCandidateWords += len(lCandidate)
        nReferenceWords += len(lReference)
        nCorrectWords, nCorrectTags, total = evaluateSentence(lCandidate, lReference)
        nTotalCorrectWords += nCorrectWords
        nTotalCorrectTags += nCorrectTags
        nTotalPrediction += total
    word_precision = float(nTotalCorrectWords) / float(nCandidateWords)
    word_recall = float(nTotalCorrectWords) / float(nReferenceWords)
    tag_precision = float(nTotalCorrectTags) / float(nCandidateWords)
    tag_recall = float(nTotalCorrectTags) / float(nReferenceWords)
    word_false_negative = nCandidateWords - nTotalCorrectWords
    tag_false_negative = nCandidateWords - nTotalCorrectTags
    word_real_negative = nTotalPrediction - nReferenceWords
    tag_real_negative = nTotalPrediction * tag_num - nReferenceWords
    word_tnr = 1 - float(word_false_negative) / float(word_real_negative)
    tag_tnr = 1 - float(tag_false_negative) / float(tag_real_negative)
    if word_precision + word_recall > 0:
        word_fmeasure = (2 * word_precision * word_recall) / (word_precision + word_recall)
    else:
        word_fmeasure = 0.00001

    if tag_precision + tag_recall == 0:
        tag_fmeasure = 0.0
    else:
        tag_fmeasure = (2 * tag_precision * tag_recall) / (tag_precision + tag_recall)
    if verbose:
        return word_precision, word_recall, word_fmeasure, tag_precision, tag_recall, tag_fmeasure, word_tnr, tag_tnr
    else:
        return word_fmeasure, tag_fmeasure


def score_boundaries(sReference, sCandidate, verbose=False):
    nTotalCorrectWords = 0
    nTotalCorrectTags = 0
    nCandidateWords = 0
    nReferenceWords = 0
    reference = readNonEmptySentenceList(sReference)
    candidate = readNonEmptySentenceList(sCandidate)
    assert len(reference) == len(candidate)
    for lReference, lCandidate in zip(reference, candidate):
        n = len(lCandidate)
        nCandidateWords += len(lCandidate) - 1
        nReferenceWords += len(lReference) - 1
        nCorrectWords, nCorrectTags = evaluateSentence_boundaries(lCandidate, lReference)
        nTotalCorrectWords += nCorrectWords
        nTotalCorrectTags += nCorrectTags
    word_precision = float(nTotalCorrectWords) / float(nCandidateWords)
    word_recall = float(nTotalCorrectWords) / float(nReferenceWords)
    tag_precision = float(nTotalCorrectTags) / float(nCandidateWords)
    tag_recall = float(nTotalCorrectTags) / float(nReferenceWords)
    if word_precision + word_recall > 0:
        word_fmeasure = (2 * word_precision * word_recall) / (word_precision + word_recall)
    else:
        word_fmeasure = 0.00001

    if tag_precision + tag_recall == 0:
        tag_fmeasure = 0.0
    else:
        tag_fmeasure = (2 * tag_precision * tag_recall) / (tag_precision + tag_recall)
    if verbose:
        return word_precision, word_recall, word_fmeasure, tag_precision, tag_recall, tag_fmeasure
    else:
        return word_fmeasure, tag_fmeasure


#================================================================
#
# Main.
#
#================================================================

if __name__ == '__main__':
   #
   # Parse command ......
   #
   opts, args = getopt.getopt(sys.argv[1:], "")
   for opt in opts:
      print opt
   if len(args) != 2:
      print g_sInformation
      sys.exit(1)
   sCandidate = args[0]
   sReference = args[1]
   boundaries = True
   if not os.path.exists(sCandidate):
      print "Candidate file %s does not exist." % sCandidate
      sys.exit(1)
   if not os.path.exists(sCandidate):
      print "Reference file %s does not exist." % sReference
      sys.exit(1)
   #
   # Compare candidate and reference
   #
   nTotalCorrectWords = 0
   nTotalCorrectTags = 0
   nCandidateWords = 0
   nReferenceWords = 0
   fReference = CPennTaggedSentenceReader(sReference); fCandidate = CPennTaggedSentenceReader(sCandidate)
   lReference = fReference.readNonEmptySentence(bIgnoreNoneTag=True); lCandidate = fCandidate.readNonEmptySentence(bIgnoreNoneTag=True)
   if boundaries:
       while lReference and lCandidate:
           n = len(lCandidate)
           nCandidateWords += len(lCandidate) - 1
           nReferenceWords += len(lReference) - 1
           # nCorrectWords, nCorrectTags = evaluateSentence(lCandidate, lReference)
           nCorrectWords, nCorrectTags = evaluateSentence_boundaries(lCandidate, lReference)
           nTotalCorrectWords += nCorrectWords
           nTotalCorrectTags += nCorrectTags
           lReference = fReference.readNonEmptySentence(bIgnoreNoneTag=True)
           lCandidate = fCandidate.readNonEmptySentence(bIgnoreNoneTag=True)
   else:
       while lReference and lCandidate:
          n=len(lCandidate)
          nCandidateWords += len(lCandidate)
          nReferenceWords += len(lReference)
          nCorrectWords, nCorrectTags = evaluateSentence(lCandidate, lReference)
          #nCorrectWords, nCorrectTags = evaluateSentence_boundaries(lCandidate, lReference)
          nTotalCorrectWords += nCorrectWords
          nTotalCorrectTags += nCorrectTags
          lReference = fReference.readNonEmptySentence(bIgnoreNoneTag=True); lCandidate = fCandidate.readNonEmptySentence(bIgnoreNoneTag=True)

   if ( lReference and not lCandidate ) or ( lCandidate and not lReference ) :
      print "Warning: the reference and the candidate consists of different number of lines!"

   word_precision = float(nTotalCorrectWords) / float(nCandidateWords)
   word_recall = float(nTotalCorrectWords) / float(nReferenceWords)
   tag_precision = float(nTotalCorrectTags) / float(nCandidateWords)
   tag_recall = float(nTotalCorrectTags) / float(nReferenceWords)
   word_fmeasure = (2*word_precision*word_recall)/(word_precision+word_recall)
   if tag_precision+tag_recall==0:
      tag_fmeasure = 0.0
   else:
      tag_fmeasure = (2*tag_precision*tag_recall)/(tag_precision+tag_recall)

   print "Word precision:", word_precision
   print "Word recall:", word_recall

   print "Tag precision:", tag_precision
   print "Tag recall:", tag_recall

   print "Word F-measure:", word_fmeasure
   print "Tag F-measure:",  tag_fmeasure


