# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:40:20 2022

@author: a
"""

role_semantic = {
    'Event_None':' the entity that is irrelevant to the event', 
    'Event_Place':' the place where the event takes place',
  #   "Adjudicator":' the judge or court',
    
    "Life:Be-Born_Person":' the person who is born',
    
    "Life:Marry_Person":' the person who are married',
    
    "Life:Divorce_Person":' the person who are divorced',
    
    "Life:Injure_Agent":' the one that enacts the harm',
    "Life:Injure_Victim":' the harmed person',
    "Life:Injure_Instrument":' the device used to inflict the harm',
    
    "Life:Die_Agent":' the killer',
    "Life:Die_Victim":' the person who died',
    "Life:Die_Instrument":' the device used to kill',
    
    "Movement:Transport_Agent":' the agent responsible for the transport event',
    "Movement:Transport_Artifact":' the person doing the traveling or the artifact being transported',
    "Movement:Transport_Vehicle":' the vehicle used to transport the person or artifact',
    "Movement:Transport_Origin":' the place where the transporting originated',
    "Movement:Transport_Destination":' the place where the transporting is directed',
    
    "Transaction:Transfer-Ownership_Buyer":' the buying agent',
    "Transaction:Transfer-Ownership_Seller":' the selling agent',
    "Transaction:Transfer-Ownership_Beneficiary":' the agent that benefits from the transaction',
    "Transaction:Transfer-Ownership_Artifact":' the item or organization that was bought or sold',
    
    "Transaction:Transfer-Money_Giver":' the donating agent',
    "Transaction:Transfer-Money_Recipient":' the recipient agent',
    "Transaction:Transfer-Money_Beneficiary":' the agent that benefits from the transfer',
    
    "Business:Start-Org_Agent":' the founder',
    "Business:Start-Org_Org":' the organization that is started',
    
    "Business:Merge-Org_Org":' the organizations that are merged',
    
    "Business:Declare-Bankruptcy_Org":' the organization declaring bankruptcy',
    
    "Business:End-Org_Org":' the organization that is ended',
    
    "Conflict:Attack_Attacker":' the attacking agent',
    "Conflict:Attack_Target":' the target of the attack',
    "Conflict:Attack_Instrument":' the instrument used in the attack',
    
    "Conflict:Demonstrate_Entity":' the demonstrating agent',
    
    "Contact:Meet_Entity":' the agents who are meeting',
    
    "Contact:Phone-Write_Entity":' the communicating agent',
    
    "Personnel:Start-Position_Person":' the employee',
    "Personnel:Start-Position_Entity":' the employer',
    
    "Personnel:End-Position_Person":' the employee',
    "Personnel:End-Position_Entity":' the employer',
    
    "Personnel:Elect_Person":' the person elected',
    "Personnel:Elect_Entity":' the voting agent',
    
    "Personnel:Nominate_Person":' the person nominated',
    "Personnel:Nominate_Agent":' the nominating agent',
    
    "Justice:Arrest-Jail_Person":' the person who is jailed or arrested',
    "Justice:Arrest-Jail_Agent":' the jailer or the arresting agent',
    
    "Justice:Release-Parole_Person":' the person who is released',
    "Justice:Release-Parole_Entity":' the former captor agent',
    
    "Justice:Trial-Hearing_Defendant":' the agent on trial',
    "Justice:Trial-Hearing_Prosecutor":' the prosecuting agent',
    "Justice:Trial-Hearing_Adjudicator":' the judge or court',
    
    "Justice:Charge-Indict_Defendant":' the agent that is indicted',
    "Justice:Charge-Indict_Prosecutor":' the agent bringing charges or executing the indictment',
    "Justice:Charge-Indict_Adjudicator":' the judge or court',
    
    "Justice:Sue_Plaintiff":' the suing agent',
    "Justice:Sue_Defendant":' the agent being sued',
    "Justice:Sue_Adjudicator":' the judge or court',
    
    "Justice:Convict_Defendant":' the convicted agent',
    "Justice:Convict_Adjudicator":' the judge or court',
    
    "Justice:Sentence_Defendant":' the agent who is sentenced',
    "Justice:Sentence_Adjudicator":' the judge or court',
    
    "Justice:Fine_Entity":' the entity that was fined',
    "Justice:Fine_Adjudicator":' the entity doing the fining',
    
    "Justice:Execute_Person":' the person executed',
    "Justice:Execute_Agent":' the agent responsible for carrying out the execution',
    
    "Justice:Extradite_Person":' the person being extradicted',
    "Justice:Extradite_Agent":' the extraditing agent',
    "Justice:Extradite_Origin":' the original location of the person being extradited',
    "Justice:Extradite_Destination":' the place where the person is extradited to',
    
    "Justice:Acquit_Defendant":' the agent being acquitted',
    "Justice:Acquit_Adjudicator":' the judge or court',
    
    "Justice:Pardon_Defendant":' the agent being pardoned',
    "Justice:Pardon_Adjudicator":' the state official who does the pardoning',
    
    "Justice:Appeal_Defendant":' the defendant',  #在数据集中未出现
#    "Justice:Appeal_Prosecutor":' the prosecuting agent', #在数据集中未出现
    "Justice:Appeal_Adjudicator":' the judge or court',
    'Justice:Appeal_Plaintiff': ' the appealing agent'}


event_role = {
    "Life:Be-Born": ["Life:Be-Born_Person", 'Event_Place'],
    "Life:Marry": ["Life:Marry_Person", 'Event_Place'],
    "Life:Divorce": ["Life:Divorce_Person", 'Event_Place'],
    "Life:Injure": ["Life:Injure_Agent", "Life:Injure_Victim", "Life:Injure_Instrument", 'Event_Place'],
    "Life:Die": ["Life:Die_Agent", "Life:Die_Victim", "Life:Die_Instrument", 'Event_Place'],
    "Movement:Transport": ["Movement:Transport_Agent", "Movement:Transport_Artifact", "Movement:Transport_Vehicle", "Movement:Transport_Origin", "Movement:Transport_Destination"],
    "Transaction:Transfer-Ownership": ["Transaction:Transfer-Ownership_Buyer", "Transaction:Transfer-Ownership_Seller", "Transaction:Transfer-Ownership_Beneficiary", "Transaction:Transfer-Ownership_Artifact", 'Event_Place'],
    "Transaction:Transfer-Money": ["Transaction:Transfer-Money_Giver", "Transaction:Transfer-Money_Recipient", "Transaction:Transfer-Money_Beneficiary", 'Event_Place'],
    "Business:Start-Org": ["Business:Start-Org_Agent", "Business:Start-Org_Org", 'Event_Place'],
    "Business:Merge-Org": ["Business:Merge-Org_Org", 'Event_Place'],
    "Business:Declare-Bankruptcy": ["Business:Declare-Bankruptcy_Org", 'Event_Place'],
    "Business:End-Org": ["Business:End-Org_Org", 'Event_Place'],
    "Conflict:Attack": ["Conflict:Attack_Attacker", "Conflict:Attack_Target", "Conflict:Attack_Instrument", 'Event_Place'],
    "Conflict:Demonstrate": ["Conflict:Demonstrate_Entity", 'Event_Place'],
    "Contact:Meet": ["Contact:Meet_Entity", 'Event_Place'],
    "Contact:Phone-Write": ["Contact:Phone-Write_Entity", 'Event_Place'],
    "Personnel:Start-Position": ["Personnel:Start-Position_Person", "Personnel:Start-Position_Entity", 'Event_Place'],
    "Personnel:End-Position": ["Personnel:End-Position_Person", "Personnel:End-Position_Entity", 'Event_Place'],
    "Personnel:Elect": ["Personnel:Elect_Person", "Personnel:Elect_Entity", 'Event_Place'],
    "Personnel:Nominate": ["Personnel:Nominate_Person", "Personnel:Nominate_Agent", 'Event_Place'],
    "Justice:Arrest-Jail": ["Justice:Arrest-Jail_Person", "Justice:Arrest-Jail_Agent", 'Event_Place'],    
    "Justice:Release-Parole": ["Justice:Release-Parole_Person", "Justice:Release-Parole_Entity", 'Event_Place'],
    "Justice:Trial-Hearing": ["Justice:Trial-Hearing_Defendant", "Justice:Trial-Hearing_Prosecutor", "Justice:Trial-Hearing_Adjudicator", 'Event_Place'],
    "Justice:Charge-Indict": ["Justice:Charge-Indict_Defendant", "Justice:Charge-Indict_Prosecutor", "Justice:Charge-Indict_Adjudicator", 'Event_Place'],
    "Justice:Sue": ["Justice:Sue_Defendant", "Justice:Sue_Plaintiff", "Justice:Sue_Adjudicator", 'Event_Place'],
    "Justice:Convict": ["Justice:Convict_Defendant", "Justice:Convict_Adjudicator", 'Event_Place'],
    "Justice:Sentence": ["Justice:Sentence_Defendant", "Justice:Sentence_Adjudicator", 'Event_Place'],
    "Justice:Fine": ["Justice:Fine_Entity", "Justice:Fine_Adjudicator", 'Event_Place'],
    "Justice:Execute": ["Justice:Execute_Person", "Justice:Execute_Agent", 'Event_Place'],
    "Justice:Extradite": ["Justice:Extradite_Agent", "Justice:Extradite_Origin", "Justice:Extradite_Destination","Justice:Extradite_Person"],
    "Justice:Acquit": ["Justice:Acquit_Defendant", "Justice:Acquit_Adjudicator", 'Event_Place'],
    "Justice:Pardon": ["Justice:Pardon_Defendant", "Justice:Pardon_Adjudicator", 'Event_Place'],
    "Justice:Appeal": ["Justice:Appeal_Adjudicator", 'Justice:Appeal_Plaintiff', "Justice:Appeal_Defendant", 'Event_Place']}

