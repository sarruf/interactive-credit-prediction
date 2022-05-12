import React, { useState } from 'react';
import styled from 'styled-components';
import DataTable from 'react-data-table-component';
import * as XLSX from 'xlsx';
import { LineChart, ResponsiveContainer, Legend, Tooltip, 
  Line, XAxis, YAxis, CartesianGrid } from 'recharts';
import './App.css';

// Styling a regular HTML input
const StyledInput = styled.input`
  display: block;
  margin: 20px 0px;
  border: 1px solid lightblue;
`;

// Creating a custom hook
function useInput(defaultValue) {
  const [value, setValue] = useState(defaultValue);
  function onChange(e) {
    setValue(e.target.value);
  }
  return {
    value,
    onChange,
  };
}

const checkboxesList = [
  'Possui carro',
  'Possui imóvel',
  'Possui telefone celular',
  'Possui telefone profissional',
  'Possui telefone fixo',
  'Possui e-mail'
];
const getDefaultCheckboxes = () =>
  checkboxesList.map(checkbox => ({
    name: checkbox,
    checked: false,
  }));
export function useCheckboxes(defaultCheckboxes) {
  const [checkboxes, setCheckboxes] = useState(
    defaultCheckboxes || getDefaultCheckboxes(),
  );
  function setCheckbox(index, checked) {
    const newCheckboxes = [...checkboxes];
    newCheckboxes[index].checked = checked;
    setCheckboxes(newCheckboxes);
  }
  return {
    setCheckbox,
    checkboxes,
  };
}
const Checkbox = styled.input`
  margin: 0px 10px 0px !important;
  cursor: pointer;
`;
const CheckboxLabel = styled.label`
  cursor: pointer;
  display: block;
  font-weight: normal;
`;
export function Checkboxes({ checkboxes, setCheckbox }) {
  return (
    <>
      {checkboxes.map((checkbox, i) => (
        <CheckboxLabel>
          <Checkbox
            type="checkbox"
            checked={checkbox.checked}
            onChange={e => {
              setCheckbox(i, e.target.checked);
            }}
          />
          {checkbox.name}
        </CheckboxLabel>
      ))}
    </>
  );
}

// Sample chart data
const pdata = [];
const pdata2 = [];

function App() {	
	const [columns, setColumns] = useState([]);
  const [data, setData] = useState([]);
	const [chartData, setChartData] = useState(pdata);
	const [clientId, setClientId] = useState("5001720");
	const [ranking, setRanking] = useState("Nenhum modelo novo foi treinado");
	const [currentModel, setCurrentModel] = useState("O modelo inicial ainda não foi retreinado");
	const [currentSimul, setCurrentSimul] = useState("Não foi simulado o crédito de nenhum indivíduo");
	const [income, setIncome] = useState(0);
	const [education, setEducation] = useState(0);
	const [marital, setMarital] = useState(0);
	const [estate, setEstate] = useState(0);
	const [occupation, setOccupation] = useState(0);
	const [gender, setGender] = useState(0);
	const [tech, setTech] = useState(0);
	const [n_meses, setMonthly] = useState(0);
	const [credit, setCredit] = useState(0);

	const inputClientId = useInput("5001720");
  const inputNFilho = useInput(2);
	const inputRendaTotal = useInput(200000);
	const inputDataNasc = useInput("01-01-1980");
	const inputDataAdm = useInput("01-01-2000");
	const inputNFamilia = useInput(4);
	
	const checkboxes = useCheckboxes();
	const incomeList = [{label:"Trabalho privado", value:0},{label:"Associação comercial",value:1},{label:"Pensionista",value:2},{label:"Servidor público",value:3},{label:"Estudante",value:4}];
	const educationList = [{label:"Ensino médio",value:0}, {label:"Ensino superior", value:1}, {label:"Ensino superior incompleto",value:2},{label:"Ensino fundamental",value:3},{label:"Pós-graduação",value:4}];
	const maritalStatusList = [{label:"Casado(a)", value:0}, {label:"Solteiro(a)", value:1}, {label:"Casado(a) no civil", value:2},{label:"Separado(a)", value:3},{label:"Viúvo(a)", value:4}];
	const estateTypeList = [{label:"Casa/apartamento próprio",value:0}, {label:"Mora com os pais",value:1}, {label:"Apartamento do Estado", value:2},{label:"Apartamento alugado", value:3},{label:"Escritório", value:4}];
	const occupationList = [{label:"Profissional de limpeza",value:0}, {label:"Profissional de cozinha",value:1}, {label:"Motorista", value:2},{label:"Profissional", value:3},{label:"Profissional de baixa qualificação", value:4},{label:"Profissional de segurança", value:5},{label:"Garçom/barman", value:6},{label:"Contador(a)", value:7},{label:"Funcionário(a) de escritório", value:8},{label:"Funcionário(a) de RH", value:9},{label:"Profissional de saúde", value:10},{label:"Funcionário(a) do serviço privado", value:11},{label:"Corretor(a) de imóveis", value:12},{label:"Vendedor(a)", value:13},{label:"Secretário(a)", value:14},{label:"Gerente", value:15},{label:"Funcionário(a) de tecnologia qualificado(a)", value:16},{label:"Funcionário(a) de TI", value:17}];
	const genderList = [{label:"Feminino",value:0}, {label:"Masculino",value:1}];
	const techList = [{label:"Regressão Logística",value:0}, {label:"Árvore de Decisão",value:1}, {label:"Random Forest",value:2}, {label:"SVM",value:3},{label:"LGBM",value:4},{label:"XGBoost",value:5},{label:"Cat Boost (leva alguns minutos)",value:6}];
	const monthlyBillList	= [{label:"1 mês de atraso",value:0}, {label:"2 meses de atraso",value:1}, {label:"3 meses de atraso",value:2}, {label:"4 meses de atraso",value:3}, {label:"5 meses de atraso",value:4}];
	const losingCreditList = [{label:"1 vez",value:0}, {label:"2 vezes",value:1}, {label:"3 vezes", value:2}, {label:"4 vezes", value:3}, {label:"5 vezes", value:4}];
	
	const handleUpdateClick = e => {
		const url = "http://localhost:5000/update"
		fetch(url)
		.then((res) => res.json())
		.then((data) => 
			setRanking(data.map(
				(dt) => {
					return <tr><td>{dt.data}</td><td>{dt.tech}</td><td>{dt.meses}</td><td>{dt.atraso}</td><td>{dt.accuracy.toFixed(4)}</td><td>{dt.precision.toFixed(4)}</td><td>{dt.recall.toFixed(4)}</td><td>{dt.f1.toFixed(4)}</td></tr>
				}
			)));
	}
	
	const handleResetClick = e => {
		const url = "http://localhost:5000/reset"
		fetch(url)
		.then(setRanking("Nenhum modelo foi treinado"));
	}
	
	const handleRetrainClick = e => {
		const url = "http://localhost:5000/retrain/" + tech + "/" + n_meses + "/" + credit
		fetch(url)
		.then((res) => res.json())
		.then((data) => {
				setCurrentModel(data.model);
		});
	}
	
	const handleSimulClick = e => {
		const url = "http://localhost:5000/simular/" + inputNFilho.value + "/" + inputRendaTotal.value + "/" +
		inputDataNasc.value + "/" + inputDataAdm.value + "/" + inputNFamilia.value + "/" + income + "/" + education +
		"/" + marital + "/" + estate + "/" + occupation + "/" + gender + "/" + checkboxes.checkboxes
		.map(checkbox => checkbox.checked).join('/')
		fetch(url)
		.then((res) => res.json())
		.then((data) => {
				setCurrentSimul(data.simul);
		});
	}
		
  // process CSV data
	const processData = dataString => {
		const dataStringLines = dataString.split(/\r\n|\n/);
		const headers = dataStringLines[0].split(/,(?![^"]*"(?:(?:[^"]*"){2})*[^"]*$)/);

		const list = [];
		for (let i = 1; i < dataStringLines.length; i++) {
			const row = dataStringLines[i].split(/,(?![^"]*"(?:(?:[^"]*"){2})*[^"]*$)/);
			if (headers && row.length === headers.length) {
			const obj = {};
			for (let j = 0; j < headers.length; j++) {
				let d = row[j];
				if (d.length > 0) {
				if (d[0] === '"')
					d = d.substring(1, d.length - 1);
				if (d[d.length - 1] === '"')
					d = d.substring(d.length - 2, 1);
				}
				if (headers[j]) {
				obj[headers[j]] = d;
				}
			}

			// remove the blank rows
			if (Object.values(obj).filter(x => x).length > 0) {
				if (obj['"ID"'] === clientId) {
					pdata2.push(obj);
				}
				
				list.push(obj);
			}
	  }
		
	}
	
	setChartData(pdata2);

	// prepare columns list from headers
	const columns = headers.map(c => ({
	  name: c,
	  selector: c,
	}));

	setData(list);
	setColumns(columns);
	}

	// handle file upload
	const handleFileUpload = e => {
		const file = e.target.files[0];
		const reader = new FileReader();
		reader.onload = (evt) => {
			/* Parse data */
			const bstr = evt.target.result;
			const wb = XLSX.read(bstr, { type: 'binary' });
			/* Get first worksheet */
			const wsname = wb.SheetNames[0];
			const ws = wb.Sheets[wsname];
			/* Convert array of arrays */
			const data = XLSX.utils.sheet_to_csv(ws, { header: 1 });
			processData(data);
		};
		reader.readAsBinaryString(file);
	}

  return (
	<div className="App">

		<h2>Ranking das métricas dos 10 melhores modelos treinados</h2>
		
		<button style={{marginRight: "10"}} onClick={() => handleUpdateClick()}>Atualizar</button>
		<button style={{marginLeft: "10"}} onClick={() => handleResetClick()}>Resetar</button> 
		<table className="table table-striped">
			<thead>
				<tr>
					<th>Data</th>
					<th>Modelo</th>
					<th>Limiar de atraso (meses)</th>
					<th>Número de atrasos</th>
					<th><strong>Acurácia</strong></th>
					<th><strong>Precisão</strong></th>
					<th><strong>Recall</strong></th>
					<th><strong>F1</strong></th>
				</tr>
			</thead>
			<tbody>
			{ranking}
			</tbody>
		</table>
		
		<br/><br/>
		
		<h2>Modificação do modelo treinado</h2>
	
		<div style={{textAlign: "left"}}>
		  Técnica utilizada: 
		  <select value={tech} onChange={(e) => setTech(e.target.value)}>
			{techList.map((tec) => {
				return <option value={tec.value}>{tec.label}</option>;
			})}
			</select>
		</div>
		<div style={{textAlign: "left"}}>
		  <strong>Critérios para exclusão do crédito:</strong>
			<div style={{textAlign: "left"}}>
				Limiar de atraso (em meses) do mal pagador:
				<select value={n_meses} onChange={(e) => setMonthly(e.target.value)}>
				{monthlyBillList.map((mon) => {
					return <option value={mon.value}>{mon.label}</option>;
				})}
				</select>
			</div>
			<div style={{textAlign: "left"}}>
				Número de vezes em que cliente ultrapassou limiar de atraso acima, para ser mal pagador:
				<select value={credit} onChange={(e) => setCredit(e.target.value)}>
				{losingCreditList.map((cre) => {
					return <option value={cre.value}>{cre.label}</option>;
				})}
				</select>
			</div>
		</div>
		<br/>
		<br/>
		<button onClick={() => handleRetrainClick()}>Retreinar</button> 
		<br/>
		<br/>
		<h4 style={{backgroundColor: "grey"}}>{currentModel}</h4>
		
		<br/><br/>
	
	
		<h2>Classificação de possível cliente</h2>
	
	  <div style={{margin: 20}}>
		<div>
		  <label style={{float: "left"}} htmlFor="n-filho">Número de filhos: </label> 
		  <StyledInput id="n-filho" placeholder="2" {...inputNFilho}/>
		</div>
		<div>
		  <label style={{float: "left"}} htmlFor="renda-total">Renda anual (em US$): </label> 
		  <StyledInput id="renda-total" placeholder="200000"			{...inputRendaTotal}/>
		</div>
		<div>
			<label style={{float: "left"}} htmlFor="data-nasc">Data de nascimento: </label> 
			<StyledInput id="data-nasc" placeholder="01/01/1980" {...inputDataNasc}/>
		</div>
		<div>
			<label style={{float: "left"}} htmlFor="data-adm">Data de admissão no emprego atual: </label> 
			<StyledInput id="data-adm" placeholder="01/01/2000" {...inputDataAdm}/>
		</div>
		<div>
			<label style={{float: "left"}} htmlFor="n-familia">Número de integrantes da família: </label> 
			<StyledInput id="n-familia" placeholder="4" {...inputNFamilia}/>
		</div>
		<div style={{textAlign: "left"}}>
		  Tipo de renda: 
		  <select value={income} onChange={(e) => setIncome(e.target.value)}>
			{incomeList.map((inc) => {
				return <option value={inc.value}>{inc.label}</option>;
			})}
			</select>
		</div>
		<div style={{textAlign: "left"}}>
		  Nível de escolaridade:
			<select value={education} onChange={(e) => setEducation(e.target.value)}>
			{educationList.map((edu) => {
				return <option value={edu.value}>{edu.label}</option>;
			})}
			</select>
		</div>
		<div style={{textAlign: "left"}}>
		  Estado civil:
		  <select value={marital} onChange={(e) => setMarital(e.target.value)}>
			{maritalStatusList.map((mar) => {
				return <option value={mar.value}>{mar.label}</option>;
			})}
			</select>
		</div>
		<div style={{textAlign: "left"}}>
			Tipo de moradia:
			<select value={estate} onChange={(e) => setEstate(e.target.value)}>
			{estateTypeList.map((mor) => {
				return <option value={mor.value}>{mor.label}</option>;
			})}
			</select>
		</div>
		<div style={{textAlign: "left"}}>
			Tipo de ocupação:
			<select value={occupation} onChange={(e) => setOccupation(e.target.value)}>
			{occupationList.map((occ) => {
				return <option value={occ.value}>{occ.label}</option>;
			})}
			</select>
		</div>
		<div style={{textAlign: "left"}}>
			Gênero:
			<select value={gender} onChange={(e) => setGender(e.target.value)}>
			{genderList.map((gen) => {
				return <option value={gen.value}>{gen.label}</option>;
			})}
			</select>
		</div>
		<div style={{textAlign: "left"}}>
			<Checkboxes {...checkboxes} style={{float: "left"}}/>
			{/*<span>
			Value: 
			{income},{education},{marital},{estate},{occupation},{gender},{tech},{monthly},{credit},
			{inputNFilho.value},{inputRendaTotal.value}, 
			{inputDataNasc.value},{inputDataAdm.value}, {inputNFamilia.value}, 
			{checkboxes.checkboxes
			  .map(checkbox => checkbox.checked)
			  .join(', ')} 
		  </span>*/}
		</div>
	</div>
	<br/>
	<button onClick={() => handleSimulClick()}>Simular crédito</button> 
	<br/>
	<br/>
	<h4 style={{backgroundColor: "grey"}}>{currentSimul}</h4>
	<br/><br/>
	
	
	<h2>Evolução do pagamento de um cliente (id={clientId})</h2>
		<p>Para encontrar os IDs de clientes, abra o banco clicando em Browse... e selecionando credit_record.csv</p>
		<p>Para visualizar a evolução de um cliente:</p>
		<ul>
			<li>Digite seu ID abaixo e clique em Atualizar</li>
			<li>Abra novamente o banco, clicando em Browse... e selecionando credit_record.csv</li>
		</ul>
	  <ResponsiveContainer width="100%" aspect={3}>
			<LineChart data={chartData} margin={{ right: 300 }}>
				<CartesianGrid />
				<XAxis dataKey='MONTHS_BALANCE' />
				<YAxis></YAxis>
				<Legend />
				<Tooltip />
				<Line dataKey="STATUS"
						stroke="red" activeDot={{ r: 8 }} />
			</LineChart>
		</ResponsiveContainer>
		<div>
			<label style={{float: "left", marginLeft: "40%"}} htmlFor="client-id">ID do cliente a exibir: </label>
			<StyledInput id="client-id" placeholder="5001720" {...inputClientId}/>
			<button onClick={() => setClientId(inputClientId.value)}>Atualizar</button> 
		</div>
		<br/><br/><br/>
	
	
	<h2>Banco de dados (CSV)</h2>
	
	  <div style={{margin: 20}}>
	    <input type="file" accept=".csv,.xlsx,.xls" onChange={handleFileUpload}/>
        <DataTable pagination highlightOnHover columns={columns} data={data}/>
	  </div>
	</div>
  );
}

export default App;
