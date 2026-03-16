1️⃣ Intent: Validate User Request

Test Queries

I recently joined the procurement team in Germany and need SAP access.

Please grant me the role ZC:P2P:PO_CREATOR________:1000.

Can you show all the SAP access I currently have?

I want to review the roles assigned to me and update them.

Can you show the access rights of my direct reports?

I just joined the finance department and need access to process goods receipts.

I am leaving the company next week, please revoke my SAP access.

Please revoke SAP access for my team member who is leaving the organization.

I want to review the access assigned to the roles I own.

As an auditor, I want to review risks across SAP access assignments.

2️⃣ Intent: Detect Roles Requested

Test Queries

What roles do my colleagues in procurement have?

I am a purchaser and need the role that allows me to create purchase orders.

Which role allows users to process goods receipts?

What roles are typically assigned to finance analysts?

Can you suggest the SAP role needed to create purchase orders?

I want the same role that my peers in the team have.

Which role should I request if I need to approve purchase orders?

What role is used for invoice posting in SAP?

3️⃣ Intent: Check for SoD (Segregation of Duties)

Test Queries

Please give me access to ZC:P2P:PO_CREATOR________:1000.

I need a role that allows me to create purchase orders.

Can you check if assigning this role will create any SoD conflicts?

Why is the role assignment for user John flagged as a risk?

Can you analyze whether my current access violates any SoD policies?

Why can't I request the purchase order creator role?

Does my existing role conflict with the purchase order approval role?

4️⃣ Intent: Elaborate Role Functions

Test Queries

Can you explain the functions of the PO Creator role?

What transactions are included in the ZC:P2P:PO_CREATOR________:1000 role?

What does the goods receipt processor role allow users to do?

Can you explain the responsibilities of a purchase order approver role?

Which SAP transactions are included in the GR Processor role?

What authorization objects are used in the purchase order creation role?

What activities are allowed under the finance invoice posting role?

5️⃣ Intent: Knowledge Graph Based Scenarios

Test Queries

Show me the access assignments of my direct reports.

I want to review the roles I am responsible for as a role owner.

Can you show the users assigned to the roles I own?

Show me the role access hierarchy for my department.

I want to review risks associated with the roles under my ownership.

As an auditor, I want to see all risky access combinations in SAP.

Provide a visual overview of SAP access risks across the organization.

6️⃣ Intent: Run a What-If Scenario

Test Queries

If I remove ME29 from the PO Creator role, will the SoD conflict disappear?

What happens if I remove the release authorization objects from this role?

If I add transaction MIGO to this role, will it cause any SoD issues?

What if I combine the GR Processor role with invoice posting access?

Will adding authorization object Z_XXXXX introduce any SoD conflicts?

If a user has both goods receipt and payment roles, will it create a risk?

Can you simulate SoD impact if I modify this role?
