from .prepfold import pfd
from .samples import downsample, normalize
import numpy as np
import presto.psr_utils as psr_utils

class pfddata(pfd):
    initialized = False
    #__counter__ = [0]
    def __init__(self, filename, align=True, centre=True):
        """
        pfddata: a wrapper class around prepfold.pfd
        
        Args:
        filename : the pfd filename, or "self", then don't try to load a file.
        Optionally: 
        align : ensure that binned data falls on max(sum profile).
                this aids in interpolation of the original data onto 
                the downsampled grid. [Default = False]
                Improved summed profile (negligible change to intervals and subband plots)
        centre : shift the feature to the phase 0.5 
                 (classifier.combinedAI.fit has a randomshift parameter which can re-randomize things)
                
        """
        if filename != "self":
            pfd.__init__(self, filename)
        self.dedisperse(DM=self.bestdm, doppler=1)
        self.adjust_period()
        #pfddata.__counter__[0] += 1
        #print pfddata.__counter__
        #print 'file initialization No.:', pfddata.__counter__[0]
        if not 'extracted_feature' in self.__dict__:
            self.extracted_feature = {}
        self.extracted_feature.update({"ratings:['period']":np.array([self.topo_p1])})


        if centre:
            mx = self.profs.sum(0).sum(0).argmax()
            nbin = self.proflen
            #number of bins from 
            noff = round(nbin/2 - mx)
            self.profs = np.roll(self.profs, noff, axis=-1)
        if align:
            #ensure downsampled grid falls bin of max(profile)
            self.align = self.profs.sum(0).sum(0).argmax()
        else:
            self.align = 0
        self.initialized = True
        

    def getdata(self, phasebins=0, freqbins=0, timebins=0, DMbins=0, intervals=0, subbands=0, bandpass=0, ratings=None):
        """
        input: feature=feature_size
        possible features:
            phasebins: summmed profile, data cube (self.profs) summed(projected) to the phase axis.
            freqbins: summed frequency profile, data cube projected to the frequency axis
            timebins: summed time profile, data cube projected to the time axis.
            DMbins: DM curves.
            intervals: the time vs phase image
            subbands: the subband vs phase image
            ratings: List of possible rating stored in the pfd file, possible values including: period, redchi2, offredchi2, avgvoverc
        usage examples:
        """
        if not 'extracted_feature' in self.__dict__:
            self.extracted_feature = {}
        profs = self.profs

        if not self.initialized:
            print('pfd not initialized.')
            self.__init__('self')

        def getsumprofs(M):
            feature = '%s:%s' % ('phasebins', M)
            if M == 0:
                return np.array([])
            if not feature in self.extracted_feature:
                data = profs.sum(0).sum(0)
                self.extracted_feature[feature]  = normalize(downsample(data,M,align=self.align).ravel())
            return self.extracted_feature[feature]
        def getfreqprofs(M):
            feature = '%s:%s' % ('freqbins', M)
            if M == 0:
                return np.array([])
            if not feature in self.extracted_feature:
                self.extracted_feature[feature] = normalize(downsample(profs.sum(1).sum(0),M).ravel())
            return self.extracted_feature[feature]
        def gettimeprofs(M):
            feature = '%s:%s' % ('timebins', M)
            if M == 0:
                return np.array([])
            if not feature in self.extracted_feature:
                self.extracted_feature[feature] = normalize(downsample(profs.sum(0).sum(1),M).ravel())
            return self.extracted_feature[feature]
        def getbandpass(M):
            feature = '%s:%s' % ('bandpass', M)
            if M == 0:
                return np.array([])
            if not feature in self.extracted_feature:
                self.extracted_feature[feature] = normalize(downsample(profs.sum(0).sum(1),M).ravel())
            return self.extracted_feature[feature]
        def getDMcurve(M): # return the normalized DM curve downsampled to M points
            feature = '%s:%s' % ('DMbins', M)
            if M == 0:
                return np.array([])
            if not feature in self.extracted_feature:
                ddm = (self.dms.max() - self.dms.min())/2.
                loDM, hiDM = (self.bestdm - ddm , self.bestdm + ddm)
                loDM = max((0, loDM)) #make sure cut off at 0 DM
                hiDM = max((ddm, hiDM)) #make sure cut off at 0 DM
                N = 100
                interp = False
                sumprofs = self.profs.sum(0)
                if not interp:
                    profs = sumprofs
                else:
                    profs = np.zeros(np.shape(sumprofs), dtype='d')
                DMs = psr_utils.span(loDM, hiDM, N)
                chis = np.zeros(N, dtype='f')
                subdelays_bins = self.subdelays_bins.copy()
                for ii, DM in enumerate(DMs):
                    subdelays = psr_utils.delay_from_DM(DM, self.barysubfreqs)
                    hifreqdelay = subdelays[-1]
                    subdelays = subdelays - hifreqdelay
                    delaybins = subdelays*self.binspersec - subdelays_bins
                    if interp:
                        interp_factor = 16
                        for jj in range(self.nsub):
                            profs[jj] = psr_utils.interp_rotate(sumprofs[jj], delaybins[jj],
                                                                zoomfact=interp_factor)
                        # Note: Since the interpolation process slightly changes the values of the
                        # profs, we need to re-calculate the average profile value
                        avgprof = (profs/self.proflen).sum()
                    else:
                        new_subdelays_bins = np.floor(delaybins+0.5)
                        for jj in range(self.nsub):
                            #profs[jj] = psr_utils.rotate(profs[jj], new_subdelays_bins[jj])
                            delay_bins = int(new_subdelays_bins[jj] % len(profs[jj]))
                            if delay_bins!=0:
                                profs[jj] = np.concatenate((profs[jj][delay_bins:], profs[jj][:delay_bins]))

                        subdelays_bins += new_subdelays_bins
                        avgprof = self.avgprof
                    sumprof = profs.sum(0)
                    chis[ii] = self.calc_redchi2(prof=sumprof, avg=avgprof)
                DMcurve = normalize(downsample(chis, M))
                self.extracted_feature[feature] = DMcurve
            return self.extracted_feature[feature]

        def greyscale(img):
            global_max = np.maximum.reduce(np.maximum.reduce(img))
            min_parts = np.minimum.reduce(img, 1)
            img = (img-min_parts[:,np.newaxis])/global_max
            return img

        def getintervals(M):
            feature = '%s:%s' % ('intervals', M)
            if M == 0:
                return np.array([])
            if not feature in self.extracted_feature:
                img = greyscale(self.profs.sum(1)) 
                #U,S,V = svd(img)
                #imshow(img)
                #m,n = img.shape
                #S = resize(S,[m,1]) * eye(m,n)
                #k = 6
                #imshow(np.dot(U[:,1:k], dot(S[1:k,1:k],V[1:k,:])))
                #show()
                #if M <= len(S):
                    #return S[:M]
                #else:
                    #while len(S) < M:
                        #np.append(S, 0.)
                    #return S
                #self.extracted_feature[feature] = normalize(downsample(img, M, align=self.align).ravel())#wrong!
                self.extracted_feature[feature] = normalize(downsample(img, M, align=self.align)).ravel()
            return self.extracted_feature[feature]

        def getsubbands(M):
            feature = '%s:%s' % ('subbands', M)
            if M == 0:
                return np.array([])
            if not feature in self.extracted_feature:
                img = greyscale(self.profs.sum(0))
                #U,S,V = svd(img)
                #if M <= len(S):
                    #return S[:M]
                #else:
                    #while len(S) < M:
                        #np.append(S, 0.)
                    #return S
                #self.extracted_feature[feature] = normalize(downsample(img, M, align=self.align).ravel())
                self.extracted_feature[feature] = normalize(downsample(img, M, align=self.align)).ravel()
            return self.extracted_feature[feature]

        def getratings(L):
            feature = '%s:%s' % ('ratings', L)
            if L == None:
                return np.array([])
            if not feature in self.extracted_feature:
                result = []
                for rating in L:
                    if rating == 'period':
                        result.append(self.topo_p1)
                    elif rating == 'redchi2':
                        result.append(self.calc_redchi2())
                    elif rating == 'varprof':
                        result.append(self.calc_varprof())
                    elif rating == 'offredchi2':
                        result.append(self.estimate_offsignal_redchi2())
                    elif rating == 'avgvoverc':
                        result.append(self.avgvoverc)
                    else:
                        result.append(self.__dict__[rating])
                self.extracted_feature[feature] = np.array(result)
            return self.extracted_feature[feature]


        
        data = np.hstack((getsumprofs(phasebins), getfreqprofs(freqbins), gettimeprofs(timebins), getbandpass(bandpass), getDMcurve(DMbins), getintervals(intervals), getsubbands(subbands), getratings(ratings)))
        return data 