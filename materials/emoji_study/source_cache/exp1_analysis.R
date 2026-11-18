rm(list = ls(all = TRUE))

library(plyr)
library(ggplot2)
library(sciplot)
library(lme4)

####################################
# Demographic data
####################################

header <- names(read.csv("demographic.csv", header = TRUE, nrows = 1, skip = 0))
demographic <- read.csv("demographic.csv", header = FALSE, col.names = header, skip = 1)
demographic <- demographic[2:nrow(demographic),]

length.unique <- function(x){length(unique(x))}

# Confirm number of participants :

length.unique(demographic$Participant.Private.ID[demographic$Question.Key=="English-status"&demographic$Response=="Yes"]) # native English speaker participants

# 80 native speaker participants

####################################
# Prepare data
####################################

header <- names(read.csv("target.csv", header = TRUE, nrows = 1, skip = 0))
target <- read.csv("target.csv", header = FALSE, col.names = header, skip = 1)
target <- target[2:nrow(target),]

header <- names(read.csv("control.csv", header = TRUE, nrows = 1, skip = 0))
control <- read.csv("control.csv", header = FALSE, col.names = header, skip = 1)
control <- control[2:nrow(control),]

data <- merge(target,control, all=TRUE)

data <- subset(data, Zone.Type=="response_slider_endValue")
data <- subset(data, Response != "")

is.numeric(data$Response)
data$Response <- as.numeric(data$Response)
is.numeric(data$Response)

# Check data :

length.unique(data$Emoji) # 5 emojis tested
length.unique(data$Environment) # 6 linguistic environments tested
length.unique(data$Inference.Type) # 4 inference types tested

data$Condition <- ifelse(data$stimuli=="stimuli1","Target","Control")
length.unique(data$Condition) # 2 conditions tested

length.unique(data$Participant.Private.ID[data$Condition=="Target"]) # 39 in target condition
length.unique(data$Participant.Private.ID[data$Condition=="Control"]) # 41 in control condition

# Rename factors and levels :

data$Condition <- ordered(data$Condition, levels=c("Target","Control"), labels=c("Target","Control"))
data$Environment <- ordered(data$Environment, levels=c("Unembedded","Might","Negation","Each","None","Exactly-one"))
data$Inference.Type <- ordered(data$Inference.Type, levels=c("Manner","Conditional","Existential","Universal"))

####################################
# Plot data
####################################

# Plot by environment :

dfc <- ddply(data, c("Participant.Private.ID","Condition","Environment","Inference.Type"),
             function(data)c(Response=mean(data$Response)))
dfc1 <- ddply(dfc, c("Condition","Environment","Inference.Type"),
              function(data)c(MEAN=mean(data$Response), SE=se(data$Response)))

boxplot.data <- aggregate(Response~Participant.Private.ID+Condition+Environment+Inference.Type, FUN=mean, data=data)

plot <- ggplot(dfc1, aes(x=Inference.Type, y=MEAN, fill=factor(Condition))) + 
  ggtitle("Endorsement of target inferences") + 
  xlab("Inference type") + ylab("% endorsement") + 
  geom_bar(width=.9, position="dodge", stat="identity", alpha=.75) +
  scale_y_continuous(limits=c(NA,NA), breaks=c(0,25,50,75,100)) +
  theme_bw() + scale_fill_manual(name="Condition", 
                                 breaks=c("Target","Control"), labels=c("Target","Control"),
                                 values=c("orangered","royalblue3")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  facet_grid(.~Environment, scales="free") +
  geom_errorbar(aes(ymin=MEAN-SE, ymax=MEAN+SE), width=.5, position=position_dodge(.9), col="black")
p <- 
  plot +
  geom_dotplot(
    data = boxplot.data, 
    aes(y=Response),
    col="black", alpha=.5, 
    binaxis='y', stackdir='center', binwidth = 1,
    position = "dodge"
  ) 
p
quartz(width=10,height=3.5)
print(p)

####################################
# Linear regression models
####################################

# For UNEMBEDDED environment, test for effect of condition (target vs. control) :

unem <- subset(data, Environment=="Unembedded")
unem$ConditionC <- ifelse(unem$Condition=="Target", -.5, .5)
unem.lm = lmer(Response~ConditionC+(1|Participant.Private.ID)+(1|Emoji), REML=FALSE, data=unem)
unem.lm_ConditionC = update(unem.lm, .~.-ConditionC)
anova(unem.lm, unem.lm_ConditionC) # X^2(1)=11, p<.001 ***

# For MIGHT environment, test for effect of condition (target vs. control) :

might <- subset(data, Environment=="Might")
might$ConditionC <- ifelse(might$Condition=="Target", -.5, .5)
might.lm = lmer(Response~ConditionC+(1|Participant.Private.ID)+(1|Emoji), REML=FALSE, data=might)
might.lm_ConditionC = update(might.lm, .~.-ConditionC)
anova(might.lm, might.lm_ConditionC) # X^2(1)=24, p<.001 ***

# For NEGATION environment, test for effect of condition (target vs. control) :

neg <- subset(data, Environment=="Negation")
neg$ConditionC <- ifelse(neg$Condition=="Target", -.5, .5)
neg.lm = lmer(Response~ConditionC+(1|Participant.Private.ID)+(1|Emoji), REML=FALSE, data=neg)
neg.lm_ConditionC = update(neg.lm, .~.-ConditionC)
anova(neg.lm, neg.lm_ConditionC) # X^2(1)=33, p<.001 *** 

# For EACH environment, test for interaction between condition and reading type :

each <- subset(data, Environment=="Each")
each$ConditionC <- ifelse(each$Condition=="Target", -.5, .5)
each$ReadingC <- ifelse(each$Inference.Type=="Existential", -.5, .5)
each.lm = lmer(Response~ConditionC*ReadingC+(1+ReadingC|Participant.Private.ID)+(1|Emoji), REML=FALSE, data=each)
each.lm_interaction = update(each.lm, .~.-ConditionC:ReadingC)
anova(each.lm, each.lm_interaction) # X^2(1)=2.3, p=.13 n.s.

# For EACH_EXISTENTIAL, test for effect of condition (target vs. control) :

eachexi <- subset(each, Inference.Type=="Existential")
eachexi.lm = lmer(Response~ConditionC+(1|Participant.Private.ID)+(1|Emoji), REML=FALSE, data=eachexi)
eachexi.lm_ConditionC = update(eachexi.lm, .~.-ConditionC)
anova(eachexi.lm, eachexi.lm_ConditionC) # X^2(1)=.005, p=.94 n.s.

# For EACH_UNIVERSAL, test for effect of condition (target vs. control) :

eachuni <- subset(each, Inference.Type=="Universal")
eachuni.lm = lmer(Response~ConditionC+(1|Participant.Private.ID)+(1|Emoji), REML=FALSE, data=eachuni)
eachuni.lm_ConditionC = update(eachuni.lm, .~.-ConditionC)
anova(eachuni.lm, eachuni.lm_ConditionC) # X^2(1)=5.5, p<.05 *

# For NONE environment, test for interaction between condition and reading type :

none <- subset(data, Environment=="None")
none$ConditionC <- ifelse(none$Condition=="Target", -.5, .5)
none$ReadingC <- ifelse(none$Inference.Type=="Existential", -.5, .5)
none.lm = lmer(Response~ConditionC*ReadingC+(1+ReadingC|Participant.Private.ID), REML=FALSE, data=none)
none.lm_interaction = update(none.lm, .~.-ConditionC:ReadingC)
anova(none.lm, none.lm_interaction) # X^2(1)=.34, p=.56 n.s.

# For NONE_EXISTENTIAL, test for effect of condition (target vs. control) :

noneexi <- subset(none, Inference.Type=="Existential")
noneexi.lm = lmer(Response~ConditionC+(1|Participant.Private.ID)+(1|Emoji), REML=FALSE, data=noneexi)
noneexi.lm_ConditionC = update(noneexi.lm, .~.-ConditionC)
anova(noneexi.lm, noneexi.lm_ConditionC) # X^2(1)=30, p<.001 ***

# For NONE_UNIVERSAL, test for effect of condition (target vs. control) :

noneuni <- subset(none, Inference.Type=="Universal")
noneuni.lm = lmer(Response~ConditionC+(1|Participant.Private.ID)+(1|Emoji), REML=FALSE, data=noneuni)
noneuni.lm_ConditionC = update(noneuni.lm, .~.-ConditionC)
anova(noneuni.lm, noneuni.lm_ConditionC) # X^2(1)=27, p<.001 ***

# For EXACTLY-ONE environment, test for interaction between condition and reading type :

exact <- subset(data, Environment=="Exactly-one")
exact$ConditionC <- ifelse(exact$Condition=="Target", -.5, .5)
exact$ReadingC <- ifelse(exact$Inference.Type=="Existential", -.5, .5)
exact.lm = lmer(Response~ConditionC*ReadingC+(1+ReadingC|Participant.Private.ID)+(1|Emoji), REML=FALSE, data=exact)
exact.lm_interaction = update(exact.lm, .~.-ConditionC:ReadingC)
anova(exact.lm, exact.lm_interaction) # X^2(1)=24, p<.001 ***

# For EXACTLY-ONE_EXISTENTIAL, test for effect of condition (target vs. control) :

exactexi <- subset(exact, Inference.Type=="Existential")
exactexi.lm = lmer(Response~ConditionC+(1|Participant.Private.ID)+(1|Emoji), REML=FALSE, data=exactexi)
exactexi.lm_ConditionC = update(exactexi.lm, .~.-ConditionC)
anova(exactexi.lm, exactexi.lm_ConditionC) # X^2(1)=7e-04, p=.98 n.s.

# For EXACTLY-ONE_UNIVERSAL, test for effect of condition (target vs. control) :

exactuni <- subset(exact, Inference.Type=="Universal")
exactuni.lm = lmer(Response~ConditionC+(1|Participant.Private.ID)+(1|Emoji), REML=FALSE, data=exactuni)
exactuni.lm_ConditionC = update(exactuni.lm, .~.-ConditionC)
anova(exactuni.lm, exactuni.lm_ConditionC) # X^2(1)=23, p<.001 ***