# DuplicatePage
# Find the page "Primary Elements," duplicate it, 
# and then put the focus back on Primary Elements by (re)making it the active page.

# Iterate over all pages in document to get a reference
# to the page titled "Primary Elements"
for page in Document.Pages:
	if page.Title == "Primary Elements":
		Document.Pages.AddDuplicate(page)
		Document.ActivePageReference = Document.Pages.Item[0]

######

# ChangeVisualizationType
# Find "Visualization 1" on the active page, 
# check if it's a scatter plot, and if it is, change it to a bar chart 
# and change its name to Visualization 2. 
# Do the reverse if the visualization is a bar chart called Visualization 2. 

# Import Visuals namespace to get access to
# VisualTypeIdentifiers enumeration
from Spotfire.Dxp.Application.Visuals import *

# Iterate over all visualizations on the active page to
# get a reference to the target visualization
for visualization in Document.ActivePageReference.Visuals:
# If the visualization is called "Visualization 1" and
# is a scatter plot, change it to a bar chart and change its name. 
	if visualization.Title == "Visualization 1" and visualization.TypeId == VisualTypeIdentifiers.ScatterPlot:
		visualization.TypeId = VisualTypeIdentifiers.BarChart
		visualization.Title = "Visualization 2"

# If the visualization is called "Visualization 2" and
# is a bar chart, change it to a scatter plot and change its name. 
	elif visualization.Title == "Visualization 2" and visualization.TypeId == VisualTypeIdentifiers.BarChart:
		visualization.TypeId = VisualTypeIdentifiers.ScatterPlot
		visualization.Title = "Visualization 1"

######

# CreateDocumentProperty
# Create a new document property called "NewProperty" and give it the value 10.

# Import Data namespace to get access to the required classes
from Spotfire.Dxp.Data import *

# Remove any existing property called "NewProperty"
try:
	DataPropertyRegistry.RemoveProperty(Document.Data.Properties, DataPropertyClass.Document, "NewProperty")
except:
	pass # i.e., do nothing if the property does not exist

# Create a property prototype
propertyPrototype = DataProperty.CreateCustomPrototype("NewProperty", 0, DataType.Integer, DataPropertyAttributes.IsVisible | DataPropertyAttributes.IsEditable | DataPropertyAttributes.IsPersistent | DataPropertyAttributes.IsPropagated)

# Instantiate the prototype
Document.Data.Properties.AddProperty(DataPropertyClass.Document, propertyPrototype)
Document.Properties["NewProperty"] = 10

# Check the property's attributes
documentProperty = DataPropertyRegistry.GetProperty(Document.Data.Properties, DataPropertyClass.Document, "NewProperty")
print documentProperty.Attributes

######

# ChangeDataAndLegend
# Change the underlying data table for a visualization, change the legend font, and hide some legend items. 
# Please note: where the script refers to

# Import Visuals namespace
from Spotfire.Dxp.Application.Visuals import *
# Import Font from System.Drawing to manipulate fonts
from System.Drawing import Font

# Change the data table to stationInventory2
for visualization in Document.ActivePageReference.Visuals:
	if visualization.TypeId == VisualTypeIdentifiers.ScatterPlot:
		visualContentObject = visualization.As[VisualContent]()
		newDataTable = Document.Data.Tables.TryGetValue("stationInventory2")[1]
		visualContentObject.Data.DataTableReference = newDataTable

# Show the legend item Data table and hide the rest
for legendItem in visualContentObject.Legend.Items:
	if legendItem.Title == "Data table":
		legendItem.Visible = True
	else:
		legendItem.Visible = False

# Change the legend font size
newFont = Font("Arial", 10)
visualContentObject.Legend.Font = newFont  

######

# AxisProperties
# Change the y-axis and marker by expressions on a scatter plot, 
# hide the x-axis scale selector, and make the x-axis label orientation vertical. 

from Spotfire.Dxp.Application.Visuals import *

# Get the visualization reference
for visualization in Document.ActivePageReference.Visuals:
	if visualization.Title == "ELEV vs. STATE":
		visualContentObject = visualization.As[VisualContent]()

# Change the y-axis and marker by expressions
visualContentObject.YAxis.Expression = "Avg([ELEV])"
visualContentObject.MarkerByAxis.Expression = "<[STATE]>"

# Hide the x-axis selector and make the labels vertical
visualContentObject.XAxis.ShowAxisSelector = False
visualContentObject.XAxis.Scale.LabelOrientation = LabelOrientation.Vertical

######

# SetupTable
# Set up a table to show only certain columns.

from Spotfire.Dxp.Application.Visuals import *

# Get the visualization reference
for visualization in Document.ActivePageReference.Visuals:
	if visualization.TypeId == VisualTypeIdentifiers.Table:
		visualContentObject = visualization.As[VisualContent]()

# Clear existing columns
visualContentObject.SortedColumns.Clear()
visualContentObject.TableColumns.Clear()

# Define columns to add
columnList = []
columnList.Add("STATE")
columnList.Add("STATION_NAME")
columnList.Add("ELEV")

# Add columns to table
for column in columnList:
	visualContentObject.TableColumns.Add(visualContentObject.Data.DataTableReference.Columns[column])

######

# AddHorizontalLine
# Add a red, dashed horizontal reference line to a bar chart, 
# give it a custom name, and display that name as a label.

from Spotfire.Dxp.Application.Visuals import *
from Spotfire.Dxp.Application.Visuals.FittingModels import *
from System.Drawing import Color

# Get the visualization reference
for visualization in Document.ActivePageReference.Visuals:
	if visualization.Title == "Elevation by Latitude":
		visualContentObject = visualization.As[VisualContent]()

dataTable = Document.Data.Tables.TryGetValue("stationInventory1")[1]

# Clear any existing lines and then add new line
visualContentObject.FittingModels.Clear()
referenceLine = visualContentObject.FittingModels.AddHorizontalLine(dataTable, "[Average Elevation]")

# Style the new line
referenceLine.Line.Color = Color.Red
referenceLine.Line.CustomDisplayName = "Average Elevation"
referenceLine.Line.LineStyle = LineStyle.Dash
referenceLine.Line.Width = 2
referenceLine.Line.Visible = True

######

# SetRegionColor
# Define colors for the values in a region column used on a color axis.

from Spotfire.Dxp.Application.Visuals import *
from System.Drawing import Color

# Get the visualization reference
for visualization in Document.ActivePageReference.Visuals:
	if visualization.Title == "GDP By Region":
		visualContentObject = visualization.As[VisualContent]()

# Set up the coloring
visualContentObject.ColorAxis.Coloring.Clear()
colorRule = visualContentObject.ColorAxis.Coloring.AddCategoricalColorRule()

# Set the color values
colorRule.Item["Asia"] = Color.FromName("Blue")
colorRule.Item["Europe"] = Color.FromName("CadetBlue")
colorRule.Item["North America"] = Color.FromName("DarkOliveGreen")
colorRule.Item["South America"] = Color.FromName("Gold")
colorRule.Item["Africa"] = Color.FromName("IndianRed")
colorRule.Item["Oceania"] = Color.FromName("Violet")

######

# ColorTop5
# Set the color for the top 5 values in a color by column.

from Spotfire.Dxp.Application.Visuals import *
from Spotfire.Dxp.Application.Visuals.ConditionalColoring import *
from System.Drawing import Color

# Get the visualization reference
for visualization in Document.ActivePageReference.Visuals:
	if visualization.Title == "GDP By Country":
		visualContentObject = visualization.As[VisualContent]()

# Clear any existing coloring
visualContentObject.ColorAxis.Coloring.Clear()

# Add the color rule
visualContentObject.ColorAxis.Coloring.AddTopNRule(5, Color.FromName("Red"))

######

# ColorTable
# Apply a color gradient to a table.

from Spotfire.Dxp.Application.Visuals import *
from Spotfire.Dxp.Application.Visuals.ConditionalColoring import *
from System.Drawing import Color

# Get the visualization reference
for visualization in Document.ActivePageReference.Visuals:
	if visualization.Title == "GDP_Data":
		visualContentObject = visualization.As[VisualContent]()

# Clear existing coloring
visualContentObject.Colorings.Clear()

# Add the coloring entity
coloring = visualContentObject.Colorings.AddNew("GDP Gradient")

# Map the column
visualContentObject.Colorings.AddMapping(CategoryKey("GDP (USD million)"), coloring)

# Add the gradient and intervals
colorRule = coloring.AddContinuousColorRule()
colorRule.IntervalMode = IntervalMode.Gradient
colorRule.Breakpoints.Add(ConditionValue.MinValue, Color.FromName("SlateBlue"))
colorRule.Breakpoints.Add(ConditionValue.MaxValue, Color.FromName("Red"))
colorRule.Breakpoints.Add(ConditionValue.MedianValue, Color.FromName("Silver"))

######

# ChangeFilterSetting
# Set a list box filter to two values.

from Spotfire.Dxp.Application.Filters import *

# Get filtering scheme reference 
filterPanel = Document.ActivePageReference.FilterPanel
filteringScheme = filterPanel.FilteringSchemeReference

# Get filter collection
dataTable = Document.Data.Tables.TryGetValue("GDP_Data")[1]
filterCollection = filteringScheme[dataTable]

# Get filter and cast it
filter = filterCollection["Region"]
filterObject = filter.As[ListBoxFilter]()

# Reset the filter and change the setting
filterObject.Reset()
filterObject.IncludeAllValues = False
filterObject.SetSelection(["Oceania", "Africa"])

######

# HideFilters

from Spotfire.Dxp.Application.Filters import *

filterPanel = Document.ActivePageReference.FilterPanel
for group in filterPanel.TableGroups:
	if group.Name == "GDP_Data":
		group.Visible = True
		if group.Expanded == True:
			group.Expanded = False
	else:
		group.Visible = False

######

# ReadTable
# Provide the user with a search box to lookup values in a table. 
# The search box is referenced in the script through a document property.

from Spotfire.Dxp.Data import *

# Define data table and cursors 
dataTable = Document.Data.Tables.TryGetValue("GDP_Data")[1]
countryCursor = DataValueCursor.CreateFormatted(dataTable.Columns["Country Name"])
gdpCursor = DataValueCursor.CreateNumeric(dataTable.Columns["GDP (USD million)"])

# Read the table
for row in dataTable.GetRows(countryCursor, gdpCursor):
	# Use Trim() method to remove any spaces
	if countryCursor.CurrentValue.Trim() == Document.Properties["SearchTerm"].Trim():
		Document.Properties["Gdp"] = gdpCursor.CurrentValue
		Document.Properties["SearchResult"] = ""
		# Exit the loop if a match is found
		break
	else:
		Document.Properties["Gdp"] = 0.0
		Document.Properties["SearchResult"] = "Not Found"

######

# SelectRows
# Find values in a data table and mark the corresponding row(s).

from Spotfire.Dxp.Data import *

# Define data table
dataTable = Document.Data.Tables.TryGetValue("GDP_Data")[1]

# Make a selection by expression
rowSelection = dataTable.Select("[Country Name] = 'Ireland' or [Country Name] = 'India'")

# Mark rows
for marking in Document.Data.Markings:
	if marking.Name == "Marking":
		marking.SetSelection(rowSelection, dataTable)

######

# GetMarkedRows
# Get marked rows and the corresponding data values. 
# Use a text area control and corresponding document property to display the results.

from Spotfire.Dxp.Data import *

dataTable = Document.Data.Tables.TryGetValue("GDP_Data")[1]

# Get marked rows as a row selection
for marking in Document.Data.Markings:
	if marking.Name == "Marking":
		rowSelection = marking.GetSelection(dataTable)

# Cast rowSelection as an index set
selectedIndexSet = rowSelection.AsIndexSet()

# Define a cursor
countryCursor = DataValueCursor.CreateFormatted(dataTable.Columns["Country Name"])

# Read the table and map the selected index set
resultString = ""
for row in dataTable.GetRows(countryCursor):
	if selectedIndexSet[row.Index] == 1:
		if resultString == "":
			resultString = countryCursor.CurrentValue
		else:
			resultString = resultString + ", " + countryCursor.CurrentValue

Document.Properties["SelectedCountries"] = resultString

######

# AddChangeCalculatedColumn
# Add a new calculated column, or if it already exists, change its expression.

from Spotfire.Dxp.Data import *

dataTable = Document.Data.Tables.TryGetValue("GDP_Data")[1]

# Define expression for calculated column
calculatedColumnExpression = "If([GDP (USD million)]>= Avg([GDP (USD million)]), 'Above average', 'Below Average')"

# Determine whether the column already exists
createColumn = True
for column in dataTable.Columns:
	if column.Name == "GDP Comparison":
		createColumn = False
		break

if createColumn == True:
	dataTable.Columns.AddCalculatedColumn("GDP Comparison", calculatedColumnExpression)
else:
	calculatedColumn = dataTable.Columns["GDP Comparison"].As[CalculatedColumn]()
	calculatedColumn.Expression = calculatedColumnExpression

######

# AddColumn
# Add a column from an external CSV file to a data table in an analysis file. 
# A text area control and associated document property is used to allow the user 
# to enter the file path to the external file.

from Spotfire.Dxp.Data import *
from Spotfire.Dxp.Data.Import import *

# Define source and target and generate ignored columns list 
targetTable = Document.Data.Tables.TryGetValue("GDP_Data")[1]
ignoredColumns = []
readerSettings = TextDataReaderSettings()
readerSettings.AddIgnoreColumn(0)
readerSettings.SetColumnName(1, "Country Name")
readerSettings.SetDataType(1, DataType.String)
readerSettings.AddIgnoreColumn(2)
readerSettings.AddIgnoreColumn(3)
readerSettings.SetColumnName(4, "G20")
readerSettings.SetDataType(4, DataType.String)
readerSettings.Separator = ","
filePath = Document.Properties["FilePath"]
sourceTable = TextFileDataSource(filePath, readerSettings)

# Define the table relationship
leftColumnSignature = DataColumnSignature("Country Name", DataType.String)
rightColumnSignature = DataColumnSignature("Country Name", DataType.String)
columnMap = {leftColumnSignature:rightColumnSignature}

# Build the column settings and add the column to the target
ignoredColumns = []
columnSettings = AddColumnsSettings(columnMap, JoinType.LeftOuterJoin, ignoredColumns)

targetTable.AddColumns(sourceTable, columnSettings)

######

# AddRows
# Add rows from one table in an analysis file to another.

from Spotfire.Dxp.Data import *
from Spotfire.Dxp.Data.Import import *

# Define source and target 
targetTable = Document.Data.Tables.TryGetValue("GDP_Data_first_half")[1]
sourceTable = DataTableDataSource(Document.Data.Tables.TryGetValue("GDP_Data_second_half")[1])
dataTable2 = Document.Data.Tables.TryGetValue("GDP_Data_second_half")[1]

# Generate ignored columns list to disregard any nonshared columns  
ignoredColumns = []
targetList = []
for column in targetTable.Columns:
	targetList.Add(column.Name)
for column in dataTable2.Columns:
	if column.Name not in targetList:
		ignoredColumns.Add(DataColumnSignature(column.Name, DataType.Undefined))

# Define the table relationships
leftColumnSignature1 = DataColumnSignature("Country Abbreviation", DataType.String)
rightColumnSignature1 = DataColumnSignature("Country Abbreviation", DataType.String)
leftColumnSignature2 = DataColumnSignature("Country Name", DataType.String)
rightColumnSignature2 = DataColumnSignature("Country Name", DataType.String)
leftColumnSignature3 = DataColumnSignature("Region", DataType.String)
rightColumnSignature3 = DataColumnSignature("Region", DataType.String)
leftColumnSignature4 = DataColumnSignature("GDP (USD million)", DataType.Integer)
rightColumnSignature4 = DataColumnSignature("GDP (USD million)", DataType.Integer)
columnMap = {leftColumnSignature1:rightColumnSignature1, leftColumnSignature2:rightColumnSignature2, leftColumnSignature3:rightColumnSignature3, leftColumnSignature4:rightColumnSignature4}

# Build the row settings and add the rows to the target
rowSettings = AddRowsSettings(columnMap, ignoredColumns)

targetTable.AddRows(sourceTable, rowSettings)
	


