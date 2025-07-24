-- Entity extraction table schema for FedProcurementData database, FBOInternalAPI schema
-- Entities are stored per OpportunityId/FileId combination
-- No dates needed - opportunities already have their own dates

-- Ensure the FBOInternalAPI schema exists
IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'FBOInternalAPI')
BEGIN
    EXEC('CREATE SCHEMA FBOInternalAPI');
END
GO

CREATE TABLE FBOInternalAPI.ExtractedEntities (
    EntityId BIGINT IDENTITY(1,1) PRIMARY KEY,
    OpportunityId UNIQUEIDENTIFIER NOT NULL,
    FileId BIGINT NULL,  -- Only populated for document-derived entities
    SourceType NVARCHAR(20) NOT NULL,  -- 'title', 'description', 'document'
    
    -- Entity attributes (all nullable, populated as found)
    Name NVARCHAR(200) NULL,
    Email NVARCHAR(200) NULL,
    PhoneNumber NVARCHAR(50) NULL,
    Title NVARCHAR(200) NULL,
    Organization NVARCHAR(300) NULL,
    
    -- Metadata
    ConfidenceScore DECIMAL(5,4) NOT NULL,
    ContextText NVARCHAR(1000) NULL,
    ExtractionMethod NVARCHAR(50) NOT NULL,
    ExtractionDate DATETIME2 DEFAULT GETDATE() NOT NULL
);
GO

-- Create indexes separately
CREATE INDEX IX_ExtractedEntities_OpportunityId ON FBOInternalAPI.ExtractedEntities (OpportunityId);
CREATE INDEX IX_ExtractedEntities_FileId ON FBOInternalAPI.ExtractedEntities (FileId);
CREATE INDEX IX_ExtractedEntities_SourceType ON FBOInternalAPI.ExtractedEntities (SourceType);
CREATE INDEX IX_ExtractedEntities_Name ON FBOInternalAPI.ExtractedEntities (Name);
CREATE INDEX IX_ExtractedEntities_Email ON FBOInternalAPI.ExtractedEntities (Email);
CREATE INDEX IX_ExtractedEntities_Phone ON FBOInternalAPI.ExtractedEntities (PhoneNumber);
CREATE INDEX IX_ExtractedEntities_Organization ON FBOInternalAPI.ExtractedEntities (Organization);
GO

-- Create unique constraint separately
CREATE UNIQUE INDEX UX_ExtractedEntities_Opp_File_Source ON FBOInternalAPI.ExtractedEntities (OpportunityId, FileId, SourceType);
GO

-- View for complete contact information
CREATE VIEW FBOInternalAPI.v_opportunity_contacts AS
SELECT 
    OpportunityId,
    FileId,
    SourceType,
    Name,
    Email,
    PhoneNumber,
    Title,
    Organization,
    ConfidenceScore,
    ContextText,
    ExtractionDate
FROM FBOInternalAPI.ExtractedEntities
WHERE ConfidenceScore >= 0.5;
GO

-- View for entities with multiple attributes
CREATE VIEW FBOInternalAPI.v_linked_entities AS
SELECT 
    OpportunityId,
    FileId,
    SourceType,
    Name,
    Email,
    PhoneNumber,
    Title,
    Organization,
    ConfidenceScore,
    ExtractionDate
FROM FBOInternalAPI.ExtractedEntities
WHERE (Name IS NOT NULL AND (Email IS NOT NULL OR PhoneNumber IS NOT NULL))
   OR (Email IS NOT NULL AND PhoneNumber IS NOT NULL)
   OR (Name IS NOT NULL AND Title IS NOT NULL)
   OR (Name IS NOT NULL AND Organization IS NOT NULL);
GO
